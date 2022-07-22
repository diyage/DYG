import torch
import torch.nn as nn
import torch.nn.functional as F
from .Tools import YOLOV2Tools


class YOLOV2Loss(nn.Module):
    def __init__(self,
                 anchor_pre_wh: tuple,
                 weight_position: float = 1.0,
                 weight_conf_has_obj: float = 1.0,
                 weight_conf_no_obj: float = 1.0,
                 weight_cls_prob: float = 1.0,
                 weight_iou_loss: float = 1.0,
                 grid_number: tuple = (13, 13),
                 image_size: tuple = (416, 416),
                 iou_th: float = 0.6,
                 loss_type: int = 1,
                 ):
        super().__init__()
        self.anchor_number = len(anchor_pre_wh)
        self.anchor_pre_wh = anchor_pre_wh
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_cls_prob = weight_cls_prob
        self.weight_iou_loss = weight_iou_loss

        self.grid_number = grid_number

        self.image_size = image_size

        self.iou_th = iou_th

        self.mse = nn.MSELoss(reduction='none')
        self.iou_loss_function = nn.SmoothL1Loss(reduction='none')

        self._grid = YOLOV2Tools.get_grid(grid_number)  # H * W * 2
        self._pre_wh = torch.tensor(anchor_pre_wh, dtype=torch.float32)  # a_n *2

        self.iteration = 0
        self.loss_type = loss_type

    def split(
            self,
            x: torch.Tensor,
    ):
        return YOLOV2Tools.split_output(x, self.anchor_number)
        # position (N, H, W, a_n, 4)
        # conf (N, H, W, a_n)
        # cls_prob  (N, H, W, a_n, cls_num)

    def txtytwth_xyxy(
            self,
            txtytwth: torch.Tensor,
    ) -> torch.Tensor:
        # offset position to abs position
        return YOLOV2Tools.xywh_to_xyxy(
            txtytwth,
            self.anchor_pre_wh,
            self.image_size,
            self.grid_number
        )

    def xyxy_txty_s_twth(
            self,
            xyxy: torch.Tensor,
    ) -> torch.Tensor:
        # abs position to offset position
        # but txty do not use arc_sigmoid
        return YOLOV2Tools.xyxy_to_xywh(
            xyxy,
            self.anchor_pre_wh,
            self.image_size,
            self.grid_number
        )

    @staticmethod
    def iou_just_wh(
            boxes0: torch.Tensor,
            boxes1: torch.Tensor,
    ):
        boxes0_wh = boxes0[..., 2:4] - boxes0[..., 0:2]  # (..., 2)

        boxes1_wh = boxes1[..., 2:4] - boxes1[..., 0:2]  # (..., 2)
        s0 = boxes0_wh[..., 0] * boxes0_wh[..., 1]
        s1 = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        inter = torch.min(boxes0_wh[..., 0], boxes1_wh[..., 0]) * torch.min(boxes0_wh[..., 1], boxes1_wh[..., 1])
        union = s0 + s1 - inter
        return inter/union
        # iou --> (N, H, W, a_n)

    def forward_1(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        # split output
        o_txtytwth, o_conf, o_cls_prob = self.split(out)
        o_conf = torch.sigmoid(o_conf)
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)

        g_xyxy, g_conf, g_cls_prob = self.split(gt)

        N, H, W, a_n, _ = o_txtytwth.shape

        # translate position
        o_xyxy = self.txtytwth_xyxy(o_txtytwth)   # N * H * W * a_n * 4

        g_txty_s_twth = self.xyxy_txty_s_twth(g_xyxy)
        o_txty_s_twth = torch.cat(
            (torch.sigmoid(o_txtytwth[..., 0:2]), o_txtytwth[..., 2:4]),
            dim=-1
        )

        # compute iou and iou mask for each box
        iou = YOLOV2Tools.compute_iou(o_xyxy.detach(), g_xyxy.detach())  # N * H * W * a_n
        assert len(iou.shape) == 4

        # compute box mask
        weight = g_conf
        positive = (g_conf > 0).float()
        ignore = (g_conf == -1.0).float()
        negative = 1.0 - positive - ignore  # N * H * W * a_n

        # position loss
        # # part one, compute the response d box position loss

        position_loss_one = self.mse(o_txty_s_twth, g_txty_s_twth).sum(dim=-1)
        position_loss_one = torch.sum(
            position_loss_one * positive * weight
        )/N

        if self.iteration < 12800:
            self.iteration += 1
            # # part two, compute all predicted box position loss
            # # regression to the anchor box

            anchor_txty_s_twth = torch.zeros_like(o_txty_s_twth).to(o_txty_s_twth.device)
            position_loss_two = self.mse(
                o_txty_s_twth,
                anchor_txty_s_twth
            ).sum(dim=-1)

            position_loss_two = torch.sum(position_loss_two) / N

        else:
            position_loss_two = 0

        # position loss
        position_loss = position_loss_one + 0.01 * position_loss_two

        # conf loss
        # # part one

        has_obj_conf_loss = self.mse(
            o_conf,
            iou.detach().clone(),
        )
        has_obj_conf_loss = torch.sum(
            has_obj_conf_loss * positive
        )/N

        # # part two

        no_obj_conf_loss = self.mse(
            o_conf,
            torch.zeros_like(o_conf).to(o_conf.device),
        )
        no_obj_conf_loss = torch.sum(
            no_obj_conf_loss * negative
        )/N

        # cls_prob loss
        cls_prob_loss = self.mse(o_cls_prob, g_cls_prob).sum(dim=-1)
        cls_prob_loss = torch.sum(
            cls_prob_loss * positive
        )/N

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_cls_prob * cls_prob_loss

        loss_dict = {
            'total_loss': loss,
            'position_loss': position_loss,
            'has_obj_conf_loss': has_obj_conf_loss,
            'no_obj_conf_loss': no_obj_conf_loss,
            'cls_prob_loss': cls_prob_loss
        }
        return loss_dict

    def forward_0(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        # split output
        o_txtytwth, o_conf, o_cls_prob = self.split(out)
        o_conf = torch.sigmoid(o_conf)
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)

        g_xyxy, g_conf, g_cls_prob = self.split(gt)

        N, H, W, a_n, _ = o_txtytwth.shape

        # translate position
        o_xyxy = self.txtytwth_xyxy(o_txtytwth)  # N * H * W * a_n * 4

        g_txty_s_twth = self.xyxy_txty_s_twth(g_xyxy)
        o_txty_s_twth = torch.cat(
            (torch.sigmoid(o_txtytwth[..., 0:2]), o_txtytwth[..., 2:4]),
            dim=-1
        )

        # compute box mask
        weight = g_conf
        positive = (g_conf > 0).float()
        ignore = (g_conf == -1.0).float()
        negative = 1.0 - positive - ignore  # N * H * W * a_n

        # position loss

        position_loss = self.mse(o_txty_s_twth, g_txty_s_twth).sum(dim=-1)
        position_loss = torch.sum(
            position_loss * positive * weight
        )/N

        # conf loss
        iou = YOLOV2Tools.compute_iou(o_xyxy, g_xyxy)
        assert len(iou.shape) == 4
        # (N, H, W, a_n)

        has_obj_conf_loss = self.mse(
            o_conf,
            iou.detach().clone(),
        )

        has_obj_conf_loss = torch.sum(
            has_obj_conf_loss * positive
        )/N

        no_obj_conf_loss = self.mse(
            o_conf,
            torch.zeros_like(o_conf).to(o_conf.device),
        )
        no_obj_conf_loss = torch.sum(
            no_obj_conf_loss * negative
        )/N

        # cls_prob loss

        cls_prob_loss = self.mse(o_cls_prob, g_cls_prob).sum(dim=-1)
        cls_prob_loss = torch.sum(
            cls_prob_loss * positive
        )/N

        # iou_loss
        iou_loss = self.iou_loss_function(iou, positive)
        iou_loss = torch.sum(
            iou_loss * positive
        )/N

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_cls_prob * cls_prob_loss + \
            self.weight_iou_loss * iou_loss

        loss_dict = {
            'total_loss': loss,
            'position_loss': position_loss,
            'has_obj_conf_loss': has_obj_conf_loss,
            'no_obj_conf_loss': no_obj_conf_loss,
            'cls_prob_loss': cls_prob_loss,
            'iou_loss': iou_loss
        }
        return loss_dict

    def forward(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        if self.loss_type == 0:
            return self.forward_0(out, gt)
        else:
            return self.forward_1(out, gt)
