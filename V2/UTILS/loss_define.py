import torch
import torch.nn as nn
from V2.UTILS.others import YOLOV2Tools
import torch.nn.functional as F


class YOLOV2Loss(nn.Module):
    def __init__(self,
                 anchor_pre_wh: tuple,
                 weight_position: float = 1.0,
                 weight_conf_has_obj: float = 1.0,
                 weight_conf_no_obj: float = 1.0,
                 weight_score: float = 1.0,
                 grid_number: tuple = (13, 13),
                 image_size: tuple = (416, 416),
                 iou_th: float = 0.6):
        super().__init__()
        self.anchor_number = len(anchor_pre_wh)
        self.anchor_pre_wh = anchor_pre_wh
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_score = weight_score

        self.grid_number = grid_number

        self.image_size = image_size

        self.iou_th = iou_th

        self.mse = nn.MSELoss()
        self._grid = YOLOV2Tools.get_grid(grid_number)  # H * W * 2
        self._pre_wh = torch.tensor(anchor_pre_wh, dtype=torch.float32)  # a_n *2

    def split(
            self,
            x: torch.Tensor,
    ):
        return YOLOV2Tools.split_output(x, self.anchor_number)

    def xywh_xyxy(
            self,
            position: torch.Tensor
    ) -> torch.Tensor:
        return YOLOV2Tools.xywh_to_xyxy(
            position,
            self.anchor_pre_wh,
            self.image_size,
            self.grid_number
        )

    def compute_iou(
            self,
            boxes0: torch.Tensor,
            boxes1: torch.Tensor,
    ):
        w0 = boxes0[..., 2] - boxes0[..., 0]  # -1
        h0 = boxes0[..., 3] - boxes0[..., 1]  # -1
        s0 = w0 * h0  # -1

        w1 = boxes1[..., 2] - boxes1[..., 0]  # -1
        h1 = boxes1[..., 3] - boxes1[..., 1]  # -1
        s1 = w1 * h1  # -1

        s = s0/s1
        mask = s > 1.0
        s[mask] = 1.0/s[mask]
        return s

    def forward(self,
                out: torch.Tensor,
                gt: torch.Tensor):
        # split output
        o_position, o_conf, o_scores = self.split(out)
        o_conf = torch.sigmoid(o_conf)
        o_scores = torch.softmax(o_scores, dim=-1)
        g_position, g_conf, g_scores = self.split(gt)

        N, H, W, a_n, _ = o_position.shape

        # translate position from xywh to xyxy
        o_xyxy = self.xywh_xyxy(o_position)  # N * H * W * a_n * 4
        g_xyxy = g_position  # N * H * W * a_n * 4

        # compute iou
        iou = self.compute_iou(o_xyxy.detach(), g_xyxy.detach())  # N * H * W * a_n
        iou_max_index = iou.max(-1)[1]  # N * H * W
        iou_max = F.one_hot(iou_max_index, a_n)   # N * H * W * a_n
        iou_smaller_than_iou_th = (iou < self.iou_th).float()  # N * H * W * a_n

        # compute box mask
        # some boxes positive , some negative and ! other are free!
        positive = (g_conf > 0).float() * iou_max  # N * H * W * a_n
        negative = (g_conf <= 0).float() + (g_conf > 0).float() * iou_smaller_than_iou_th  # N * H * W * a_n
        positive_mask = positive.bool()
        negative_mask = negative.bool()

        # position loss
        # # part one, compute the response d box position loss
        mask = positive_mask.unsqueeze(-1).expand_as(o_xyxy)  # N * H * W * a_n * 4
        position_loss_one = self.mse(o_xyxy[mask], g_xyxy[mask])
        # # part two, compute the not response d box position loss, regression to the anchor box
        mask = negative_mask.unsqueeze(-1).expand_as(o_xyxy)  # N * H * W * a_n * 4
        anchor_txy = torch.empty(size=(*o_xyxy.shape[: -1], 2)).fill_(-torch.inf)  # N * H * W * a_n * 2
        anchor_twh = torch.zeros_like(anchor_txy)  # N * H * W * a_n * 2

        anchor_position = torch.cat((anchor_txy, anchor_twh), dim=-1).to(o_position.device)  # N * H * W * a_n * 4
        anchor_xyxy = self.xywh_xyxy(anchor_position)  # N * H * W * a_n * 2

        position_loss_two = self.mse(o_xyxy[mask], anchor_xyxy[mask])
        # position loss
        position_loss = position_loss_one + position_loss_two

        # conf loss

        mask = positive_mask  # N * H * W * a_n
        o_conf_mask = o_conf[mask]
        g_conf_mask = g_conf[mask]
        has_obj_conf_loss = self.mse(o_conf_mask, g_conf_mask)

        mask = negative_mask  # N * H * W * a_n
        o_conf_mask = o_conf[mask]
        no_obj_conf_loss = self.mse(
            o_conf_mask,
            torch.zeros_like(o_conf_mask).to(o_conf_mask.device)
        )

        # score loss
        mask = positive_mask.unsqueeze(-1).expand_as(o_scores)  # N * H * W * a_n * kind_number

        o_scores_mask = o_scores[mask]
        g_scores_mask = g_scores[mask]

        score_loss = self.mse(o_scores_mask, g_scores_mask)

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_score * score_loss

        return loss
