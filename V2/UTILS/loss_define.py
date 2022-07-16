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
                 image_size: tuple = (416, 416)):
        super().__init__()
        self.anchor_number = len(anchor_pre_wh)
        self.anchor_pre_wh = anchor_pre_wh
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_score = weight_score

        self.grid_number = grid_number

        self.image_size = image_size

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

        # translate position from xywh to xyxy(compare to grid)
        o_xyxy = self.xywh_xyxy(o_position)  # N * H * W * a_n * 4
        g_xyxy = g_position  # N * H * W * a_n * 4

        # compute iou
        iou = self.compute_iou(o_xyxy.detach(), g_xyxy.detach())  # N * H * W * a_n
        iou_max_index = iou.max(-1)[1]  # N * H * W
        iou_max = F.one_hot(iou_max_index, a_n)   # N * H * W * a_n
        # there are a_n anchors but just one response(others not response) one-hot

        # compute response mask
        has_obj_response_mask = (g_conf > 0).float()  # N * H * W * a_n
        has_obj_response_mask = has_obj_response_mask * iou_max  # N * H * W * a_n
        # there are a_n anchors but just one response(others not response)
        has_obj_mask = has_obj_response_mask.bool()

        no_obj_response_mask = 1.0 - has_obj_response_mask  # N * H * W * a_n
        no_obj_mask = no_obj_response_mask.bool()
        # position loss
        mask = has_obj_mask.unsqueeze(-1).expand_as(o_xyxy)
        position_loss = self.mse(o_xyxy[mask], g_xyxy[mask])

        mask = no_obj_mask.unsqueeze(-1).expand_as(o_xyxy)

        anchor_txy = torch.empty(size=(*o_xyxy.shape[: -1], 2)).fill_(-torch.inf)

        anchor_twh = torch.zeros_like(anchor_txy)

        anchor_position = torch.cat((anchor_txy, anchor_twh), dim=-1).to(o_position.device)
        anchor_xyxy = self.xywh_xyxy(anchor_position)

        position_loss += self.mse(o_xyxy[mask], anchor_xyxy[mask])

        # conf loss
        mask = has_obj_mask
        has_obj_conf_loss = self.mse(o_conf[mask], g_conf[mask])

        mask = no_obj_mask
        o_conf_mask = o_conf[mask]
        no_obj_conf_loss = self.mse(o_conf_mask,
                                    torch.zeros_like(o_conf_mask).to(o_conf_mask.device))

        # score loss

        mask = has_obj_mask.unsqueeze(-1).expand_as(o_scores)

        score_loss = self.mse(o_scores[mask], g_scores[mask])

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_score * score_loss

        return loss
