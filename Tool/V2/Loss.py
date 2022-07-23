import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # [B, num_anchor * C, H, W] -> [B, H, W, num_anchor * C] -> [B, H*W, num_anchor*C]
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(B, H * W, abC)

        # 从pred中分离出objectness预测、类别class预测、bbox的txtytwth预测
        # [B, H*W*num_anchor, 1]
        conf_pred = prediction[:, :, :1 * self.num_anchors].contiguous().view(B, H * W * self.num_anchors, 1)
        # [B, H*W, num_anchor, num_cls]
        cls_pred = prediction[:, :, 1 * self.num_anchors: (1 + self.num_classes) * self.num_anchors].contiguous().view(
            B, H * W * self.num_anchors, self.num_classes)
        # [B, H*W, num_anchor, 4]
        txtytwth_pred = prediction[:, :, (1 + self.num_classes) * self.num_anchors:].contiguous()

        txtytwth_pred = txtytwth_pred.view(B, H * W, self.num_anchors, 4)
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
            is_target: bool = False,
    ):
        return YOLOV2Tools.split_output(
            x,
            self.anchor_number,
            is_target=is_target
        )

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
        conf_loss, cls_prob_loss, position_loss, iou_loss = self.loss(
            out,
            gt
        )

        loss = conf_loss + cls_prob_loss + position_loss

        loss_dict = {
            'total_loss': loss,
            'position_loss': position_loss,
            'conf_loss': conf_loss,
            'cls_prob_loss': cls_prob_loss,
            'iou_loss': iou_loss,
        }
        return loss_dict

    def loss(self,
             out,
             gt,
             ):
        # split output
        out_split_dict = self.split(out, is_target=False)
        o_txtytwth = out_split_dict.get('position')[0]

        o_xyxy = self.txtytwth_xyxy(o_txtytwth) / self.image_size[0]  # scaled in (0, 1)
        o_xyxy = o_xyxy.detach().clone()  # be careful !!!

        gt_split_dict = self.split(gt, is_target=True)

        g_cls_ind = gt_split_dict.get('cls_ind')
        g_weight = gt_split_dict.get('weight')

        g_txty_s_twth = gt_split_dict.get('position')[0]
        g_xyxy = gt_split_dict.get('position')[1]  # scaled in (0, 1)

        pred_conf = out_split_dict.get('conf')
        pred_cls = out_split_dict.get('cls_prob')
        pred_txty = o_txtytwth[..., 0:2]
        pred_twth = o_txtytwth[..., 2:4]
        pred_iou = YOLOV2Tools.compute_iou(o_xyxy, g_xyxy)

        # 损失函数
        conf_loss_function = MSEWithLogitsLoss(reduction='mean')
        cls_loss_function = nn.CrossEntropyLoss(reduction='none')
        txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
        twth_loss_function = nn.MSELoss(reduction='none')
        iou_loss_function = nn.SmoothL1Loss(reduction='none')

        # 标签
        gt_conf = pred_iou.detach().clone()  # be careful !!!
        gt_obj = gt_split_dict.get('conf')  # be careful !!!
        gt_cls = g_cls_ind.long()
        gt_txty = g_txty_s_twth[..., 0:2]
        gt_twth = g_txty_s_twth[..., 2:4]
        gt_box_scale_weight = g_weight
        gt_iou = (gt_box_scale_weight > 0.).float()
        gt_mask = (gt_box_scale_weight > 0.).float()

        batch_size = pred_conf.size(0)
        # 置信度损失
        conf_loss = conf_loss_function(pred_conf, gt_conf, gt_obj)

        # 类别损失
        p_cls = pred_cls.reshape(-1, pred_cls.shape[-1])
        g_cls = gt_cls.view(-1, )
        cls_loss = torch.sum(cls_loss_function(p_cls, g_cls) * gt_mask.reshape(-1, )) / batch_size

        # 边界框的位置损失
        txty_loss = torch.sum(
            torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
        twth_loss = torch.sum(
            torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight * gt_mask) / batch_size
        bbox_loss = txty_loss + twth_loss

        # iou 损失
        iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

        return conf_loss, cls_loss, bbox_loss, iou_loss

    def forward_0(
            self,
            out: torch.Tensor,
            gt: torch.Tensor
    ):

        N = out.shape[0]
        # split output
        out_split_dict = self.split(out, is_target=False)
        o_txtytwth = out_split_dict.get('position')[0]
        o_txty_s_twth = torch.cat(
            (torch.sigmoid(o_txtytwth[..., 0:2]), o_txtytwth[..., 2:4]),
            dim=-1
        )
        o_xyxy = self.txtytwth_xyxy(o_txtytwth)  # not scaled

        o_conf = out_split_dict.get('conf')
        o_conf = torch.clamp(torch.sigmoid(o_conf), 1e-4, 1.0 - 1e-4)

        o_cls_prob = out_split_dict.get('cls_prob')
        o_cls_prob = torch.softmax(o_cls_prob, dim=-1)
        # split gt
        gt_split_dict = self.split(gt, is_target=True)
        g_weight = gt_split_dict.get('conf')
        g_cls_prob = gt_split_dict.get('cls_prob')

        g_xyxy = gt_split_dict.get('position')[1]  # not scaled
        g_txty_s_twth = self.xyxy_txty_s_twth(g_xyxy)
        # compute
        # compute box mask
        weight = g_weight
        positive = (weight > 0).float()
        ignore = (weight == -1.0).float()
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
        #
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
