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

        self.grid_number = grid_number

        self.image_size = image_size

        self.iou_th = iou_th

        self.mse = nn.MSELoss()
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

    def xywh_xyxy(
            self,
            position: torch.Tensor,
            is_target: bool = False
    ) -> torch.Tensor:
        if is_target:
            # gt position is already abs position
            return position/self.image_size[0]
        else:
            # offset position to abs position
            # (but it is scaled on image size what we need is scaled on [0, 1])
            return YOLOV2Tools.xywh_to_xyxy(
                position,
                self.anchor_pre_wh,
                self.image_size,
                self.grid_number
            )/self.image_size[0]

    @staticmethod
    def iou_1(
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
        o_position, o_conf, o_cls_prob = self.split(out)
        o_conf = torch.sigmoid(o_conf)
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)

        g_position, g_conf, g_cls_prob = self.split(gt)

        N, H, W, a_n, _ = o_position.shape

        # translate position from xywh to xyxy
        o_xyxy = self.xywh_xyxy(o_position, is_target=False)   # N * H * W * a_n * 4
        g_xyxy = self.xywh_xyxy(g_position, is_target=True)  # N * H * W * a_n * 4

        # compute iou and iou mask for each box
        iou = self.iou_1(o_xyxy.detach(), g_xyxy.detach())  # N * H * W * a_n
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

        if self.iteration < 12800:
            self.iteration += 1
            # # part two, compute all predicted box position loss
            # # regression to the anchor box

            anchor_txy = torch.empty(size=(*o_xyxy.shape[: -1], 2)).fill_(-torch.inf)  # N * H * W * a_n * 2
            anchor_twh = torch.zeros_like(anchor_txy)  # N * H * W * a_n * 2

            anchor_position = torch.cat((anchor_txy, anchor_twh), dim=-1).to(o_position.device)  # N * H * W * a_n * 4
            # anchor position is not the ground truth, it is offset position.
            anchor_xyxy = self.xywh_xyxy(anchor_position, is_target=False)  # N * H * W * a_n * 2

            position_loss_two = self.mse(o_xyxy, anchor_xyxy)
        else:
            position_loss_two = 0

        # position loss
        position_loss = position_loss_one + position_loss_two

        # conf loss
        # # part one
        mask = positive_mask  # N * H * W * a_n
        has_obj_conf_loss = self.mse(o_conf[mask], iou[mask].detach().clone())
        # # part two
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

    @staticmethod
    def iou_0(bboxes_a, bboxes_b):
        """
            bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
            bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
        """
        tl = torch.max(bboxes_a[..., :2], bboxes_b[..., :2])
        br = torch.min(bboxes_a[..., 2:], bboxes_b[..., 2:])
        area_a = torch.prod(bboxes_a[..., 2:] - bboxes_a[..., :2], dim=-1)
        area_b = torch.prod(bboxes_b[..., 2:] - bboxes_b[..., :2], dim=-1)

        en = (tl < br).type(tl.type()).prod(dim=-1)
        area_i = torch.prod(br - tl, dim=-1) * en  # * ((tl < br).all())
        return area_i / (area_a + area_b - area_i)

    def forward_0(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        # split output
        o_position, o_conf, o_cls_prob = self.split(out)
        o_conf = torch.sigmoid(o_conf)
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)

        g_position, g_conf, g_cls_prob = self.split(gt)

        N, H, W, a_n, _ = o_position.shape

        # translate position from xywh to xyxy
        o_xyxy = self.xywh_xyxy(o_position, is_target=False)  # N * H * W * a_n * 4
        g_xyxy = self.xywh_xyxy(g_position, is_target=True)  # N * H * W * a_n * 4

        # compute box mask
        positive = (g_conf > 0).float()  # N * H * W * a_n
        negative = (g_conf <= 0).float()  # N * H * W * a_n
        positive_mask = positive.bool()
        negative_mask = negative.bool()

        # position loss

        mask = positive_mask.unsqueeze(-1).expand_as(o_xyxy)  # N * H * W * a_n * 4
        position_loss = self.mse(o_xyxy[mask], g_xyxy[mask])

        # conf loss
        iou = self.iou_(o_xyxy, g_xyxy)
        assert len(iou.shape) == 4
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

    def forward(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        if self.loss_type == 0:
            return self.forward_0(out, gt)
        else:
            return self.forward_1(out, gt)
