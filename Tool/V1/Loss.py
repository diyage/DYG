import torch
import torch.nn as nn
import torch.nn.functional as F
from .Tools import YOLOV1Tools


class YOLOV1Loss(nn.Module):
    def __init__(self,
                 weight_position: float = 1.0,
                 weight_conf_has_obj: float = 1.0,
                 weight_conf_no_obj: float = 1.0,
                 weight_cls_prob: float = 1.0,
                 grid_number: tuple = (7, 7),
                 image_size: tuple = (448, 448),
                 iou_th: float = 0.5):
        super().__init__()

        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_cls_prob = weight_cls_prob

        self.grid_number = grid_number

        self.image_size = image_size

        self.iou_th = iou_th

        self.mse = nn.MSELoss()
        self._grid = YOLOV1Tools.get_grid(grid_number)  # H * W * 2

    def split(
            self,
            x: torch.Tensor,
    ):
        return YOLOV1Tools.split_output(
            x
        )
        # a_n is 2
        # position (N, H, W, a_n, 4)
        # conf (N, H, W, a_n)
        # cls_prob  (N, H, W, a_n, cls_num)

    def xywh_xyxy(
            self,
            position: torch.Tensor
    ) -> torch.Tensor:
        return YOLOV1Tools.xywh_to_xyxy(
            position,
            self.image_size,
            self.grid_number
        )

    def compute_iou(
            self,
            boxes0: torch.Tensor,
            boxes1: torch.Tensor,
    ):
        # boxes (N, H, W, a_n, 4)
        return YOLOV1Tools.compute_iou(
            boxes0,
            boxes1
        )
        # iou --> (N, H, W, a_n)

    def forward(self,
                out: torch.Tensor,
                gt: torch.Tensor):
        # split output
        o_position, o_conf, o_cls_prob = self.split(out)

        o_position = torch.sigmoid(o_position)
        o_conf = torch.sigmoid(o_conf)
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)

        g_position, g_conf, g_cls_prob = self.split(gt)

        N, H, W, a_n, _ = o_position.shape

        # translate position from xywh to xyxy
        o_xyxy = self.xywh_xyxy(o_position)/self.image_size[0]  # N * H * W * a_n * 4
        g_xyxy = g_position/self.image_size[0]  # N * H * W * a_n * 4

        # compute iou
        iou = self.compute_iou(o_xyxy, g_xyxy)  # N * H * W * a_n
        iou_max_index = iou.argmax(-1)  # (N, H, W)
        iou_max = F.one_hot(iou_max_index, iou.shape[-1])

        # compute box mask
        positive = (g_conf > 0).float() * iou_max  # N * H * W * a_n
        negative = 1.0 - positive # N * H * W * a_n
        positive_mask = positive.bool()
        negative_mask = negative.bool()

        # position loss
        mask = positive_mask.unsqueeze(-1).expand_as(o_xyxy)  # N * H * W * a_n * 4
        position_loss = self.mse(o_xyxy[mask], g_xyxy[mask])

        # conf loss
        # (N, H, W, a_n)
        mask = positive_mask  # N * H * W * a_n
        has_obj_conf_loss = self.mse(o_conf[mask], iou[mask].detach().clone())

        mask = negative_mask  # N * H * W * a_n
        masked_o_conf = o_conf[mask]
        masked_g_conf = torch.zeros_like(masked_o_conf).to(masked_o_conf.device)
        no_obj_conf_loss = self.mse(
            masked_o_conf,
            masked_g_conf,
        )

        # cls_prob loss
        mask = positive_mask.unsqueeze(-1).expand_as(o_cls_prob)  # N * H * W * a_n * kind_number
        cls_prob_loss = self.mse(o_cls_prob[mask], g_cls_prob[mask])

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_cls_prob * cls_prob_loss

        return loss, position_loss, has_obj_conf_loss, no_obj_conf_loss, cls_prob_loss
