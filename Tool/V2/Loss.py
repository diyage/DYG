import torch
import torch.nn as nn
from Tool.BaseTools import BaseLoss
from .Tools import YOLOV2Tools


class RightLoss(nn.Module):
    def __init__(
            self,
            anchor_pre_wh: tuple,
            kinds_number: int = 20,
            grid_number: tuple = (13, 13),
            image_size: tuple = (416, 416),
            device: str = 'cpu'
    ):
        super().__init__()
        self.num_anchors = len(anchor_pre_wh)
        self.num_classes = kinds_number
        self.stride = image_size[0] // grid_number[0]
        self.device = device
        self.anchor_size = torch.tensor(anchor_pre_wh)
        self.input_size = image_size[0]
        self.grid_cell, self.all_anchor_wh = self.create_grid(image_size[0])

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # 生成G矩阵
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs*ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def decode_xywh(self, txtytwth_pred):
        """将txtytwth预测换算成边界框的中心点坐标和宽高 \n
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                xywh_pred : [B, H*W*anchor_n, 4] \n
        """
        B, HW, ab_n, _ = txtytwth_pred.size()
        # b_x = sigmoid(tx) + gride_x
        # b_y = sigmoid(ty) + gride_y
        xy_pred = torch.sigmoid(txtytwth_pred[:, :, :, :2]) + self.grid_cell
        # b_w = anchor_w * exp(tw)
        # b_h = anchor_h * exp(th)
        wh_pred = torch.exp(txtytwth_pred[:, :, :, 2:]) * self.all_anchor_wh
        # [H*W, anchor_n, 4] -> [H*W*anchor_n, 4]
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW*ab_n, 4) * self.stride

        return xywh_pred

    def decode_boxes(self, txtytwth_pred, requires_grad=False):
        """将txtytwth预测换算成边界框的左上角点坐标和右下角点坐标 \n
            Input: \n
                txtytwth_pred : [B, H*W, anchor_n, 4] \n
            Output: \n
                x1y1x2y2_pred : [B, H*W*anchor_n, 4] \n
        """
        # 获得边界框的中心点坐标和宽高
        xywh_pred = self.decode_xywh(txtytwth_pred)

        # 将中心点坐标和宽高换算成边界框的左上角点坐标和右下角点坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred, requires_grad=requires_grad)
        x1y1x2y2_pred[:, :, 0] = (xywh_pred[:, :, 0] - xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 1] = (xywh_pred[:, :, 1] - xywh_pred[:, :, 3] / 2)
        x1y1x2y2_pred[:, :, 2] = (xywh_pred[:, :, 0] + xywh_pred[:, :, 2] / 2)
        x1y1x2y2_pred[:, :, 3] = (xywh_pred[:, :, 1] + xywh_pred[:, :, 3] / 2)

        return x1y1x2y2_pred

    def loss(self, pred_conf, pred_cls, pred_txtytwth, pred_iou, label):
        # 损失函数
        conf_loss_function = MSEWithLogitsLoss(reduction='mean')
        cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        twth_loss_function = nn.MSELoss(reduction='none')
        iou_loss_function = nn.SmoothL1Loss(reduction='none')

        # 预测
        pred_conf = pred_conf[:, :, 0]
        pred_cls = pred_cls.permute(0, 2, 1)
        pred_txty = pred_txtytwth[:, :, :2]
        pred_twth = pred_txtytwth[:, :, 2:]
        pred_iou = pred_iou[:, :, 0]

        # 标签
        gt_conf = label[:, :, 0].float()
        gt_obj = label[:, :, 1].float()
        gt_cls = label[:, :, 2].long()
        gt_txty = label[:, :, 3:5].float()
        gt_twth = label[:, :, 5:7].float()
        gt_box_scale_weight = label[:, :, 7]
        gt_iou = (gt_box_scale_weight > 0.).float()
        gt_mask = (gt_box_scale_weight > 0.).float()

        batch_size = pred_conf.size(0)
        # 置信度损失
        conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)

        # 类别损失
        cls_loss = torch.sum(cls_loss_function(pred_cls, gt_cls) * gt_mask) / batch_size

        # 边界框的位置损失
        txty_loss = torch.sum(
            torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
        twth_loss = torch.sum(
            torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
        bbox_loss = txty_loss + twth_loss

        # iou 损失
        iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

        return conf_loss, cls_loss, bbox_loss, iou_loss

    def forward(self, prediction, target):
        # 预测
        B, abC, H, W = prediction.size()
        target = target.view(B, H * W * self.num_anchors, -1)

        res_dict = YOLOV2Tools.split_predict(
            prediction,
            self.num_anchors,
        )

        txtytwth_pred = res_dict.get('position')[0].view(B, H * W, self.num_anchors, 4)
        conf_pred = res_dict.get('conf').view(B, H * W * self.num_anchors, 1)
        cls_pred = res_dict.get('cls_prob').view(B, H * W * self.num_anchors, self.num_classes)

        # decode bbox
        x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
        x1y1x2y2_gt = target[:, :, 7:].contiguous().view(-1, 4)

        # 计算预测框和真实框之间的IoU
        iou_pred = YOLOV2Tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

        # 将IoU作为置信度的学习目标
        with torch.no_grad():
            gt_conf = iou_pred.clone()

        txtytwth_pred = txtytwth_pred.view(B, H * W * self.num_anchors, 4)
        # 将IoU作为置信度的学习目标
        # [obj, cls, txtytwth, x1y1x2y2] -> [conf, obj, cls, txtytwth]
        target = torch.cat([gt_conf, target[:, :, :7]], dim=2)

        # 计算损失
        conf_loss, cls_loss, bbox_loss, iou_loss = self.loss(
            pred_conf=conf_pred,
            pred_cls=cls_pred,
            pred_txtytwth=txtytwth_pred,
            pred_iou=iou_pred,
            label=target,
            )

        return {
           'total_loss': conf_loss + cls_loss + bbox_loss,
           'position_loss': bbox_loss,
           'conf_loss': conf_loss,
           'cls_prob_loss': cls_loss,
           'iou_loss': iou_loss,
        }


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSEWithLogitsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, targets, mask):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)

        # 被忽略的先验框的mask都是-1，不参与loss计算
        pos_id = (mask == 1.0).float()
        neg_id = (mask == 0.0).float()
        pos_loss = pos_id * (inputs - targets)**2
        neg_loss = neg_id * (inputs - 0)**2
        loss = 5.0*pos_loss + 1.0*neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = torch.sum(loss) / batch_size

            return loss

        else:
            return loss


class YOLOV2Loss(BaseLoss):
    def __init__(self,
                 anchor_pre_wh: tuple,
                 weight_position: float = 1.0,
                 weight_conf_has_obj: float = 1.0,
                 weight_conf_no_obj: float = 1.0,
                 weight_cls_prob: float = 1.0,
                 weight_iou_loss: float = 1.0,
                 grid_number: tuple = (13, 13),
                 image_size: tuple = (416, 416),
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

        self.mse = nn.MSELoss(reduction='none')
        self.iou_loss_function = nn.SmoothL1Loss(reduction='none')

        self._grid = YOLOV2Tools.get_grid(grid_number)  # H * W * 2
        self._pre_wh = torch.tensor(anchor_pre_wh, dtype=torch.float32)  # a_n *2

        self.iteration = 0
        self.loss_type = loss_type

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

        return YOLOV2Tools.xyxy_to_xywh(
            xyxy,
            self.anchor_pre_wh,
            self.image_size,
            self.grid_number
        )

    def forward_1(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        loss_func = RightLoss(
            self.anchor_pre_wh,
            device=out.device,
        )
        return loss_func(out, gt)

    def forward_0(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):
        N = out.shape[0]
        # split output
        pre_res_dict = YOLOV2Tools.split_predict(
            out,
            self.anchor_number,
        )
        pre_txtytwth = pre_res_dict.get('position')[0]  # (N, H, W, a_n, 4)
        pre_xyxy = self.txtytwth_xyxy(pre_txtytwth)  # scaled on image

        pre_conf = torch.sigmoid(pre_res_dict.get('conf'))  # (N, H, W, a_n)
        pre_cls_prob = torch.softmax(pre_res_dict.get('cls_prob'), dim=-1)  # (N, H, W, a_n, kinds_number)
        pre_txty_s_twth = torch.cat(
            (torch.sigmoid(pre_txtytwth[..., 0:2]), pre_txtytwth[..., 2:4]),
            dim=-1
        )

        # split target
        gt_res_dict = YOLOV2Tools.split_target(
            gt,
            self.anchor_number,
        )
        gt_xyxy = gt_res_dict.get('position')[1]  # (N, H, W, a_n, 4) scaled on image
        gt_txty_s_twth = self.xyxy_txty_s_twth(gt_xyxy)
        gt_conf_and_weight = gt_res_dict.get('conf')  # (N, H, W, a_n)
        # gt_conf = (gt_conf_and_weight > 0).float()
        gt_weight = gt_conf_and_weight
        gt_cls_prob = gt_res_dict.get('cls_prob')

        # get mask
        positive = (gt_weight > 0).float()
        ignore = (gt_weight == -1.0).float()
        negative = 1.0 - positive - ignore

        # compute loss
        # position loss
        temp = self.mse(pre_txty_s_twth, gt_txty_s_twth).sum(dim=-1)
        position_loss = torch.sum(
            temp * positive * gt_weight
        ) / N

        # conf loss
        # compute iou
        iou = YOLOV2Tools.compute_iou(pre_xyxy, gt_xyxy)
        iou = iou.detach()  # (N, H, W, a_n) and no grad!

        # has obj/positive loss
        temp = self.mse(pre_conf, iou)
        has_obj_loss = torch.sum(
            temp * positive
        ) / N

        # no obj/negative loss
        temp = self.mse(pre_conf, torch.zeros_like(pre_conf).to(pre_conf.device))
        no_obj_loss = torch.sum(
            temp * negative
        ) / N

        # cls prob loss
        temp = self.mse(pre_cls_prob, gt_cls_prob).sum(dim=-1)
        cls_prob_loss = torch.sum(
            temp * positive
        ) / N

        # total loss
        total_loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_loss + \
            self.weight_conf_no_obj * no_obj_loss + \
            self.weight_cls_prob * cls_prob_loss

        loss_dict = {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'has_obj_loss': has_obj_loss,
            'no_obj_loss': no_obj_loss,
            'cls_prob_loss': cls_prob_loss
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
