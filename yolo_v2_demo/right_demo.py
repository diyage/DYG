import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_v2_demo.utils.get_yov2_resnet50 import Conv, reorg_layer
from yolo_v2_demo.utils.backbone import *
import numpy as np
import tools
from Tool.V2 import *
from Tool.BaseTools import get_voc_data_loader
from tqdm import tqdm


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
        iou_pred = tools.iou_score(x1y1x2y2_pred, x1y1x2y2_gt).view(B, -1, 1)

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


class YOLOv2(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.6,
                 anchor_size=None):
        super(YOLOv2, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = 0.1
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size)
        self.num_anchors = len(anchor_size)
        self.stride = 32
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

        # 主干网络：resnet50
        self.backbone = resnet50(pretrained=trainable)

        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(2048, 1024, k=1),
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        # 融合高分辨率的特征信息
        self.route_layer = Conv(1024, 128, k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(1024 + 128 * 4, 1024, k=3, p=1)

        # 预测曾
        self.pred = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), 1)

        self.loss_func = RightLoss(
            anchor_size,
            num_classes,
            device=self.device
        )

    def create_grid(self, input_size):
        w, h = input_size, input_size
        # 生成G矩阵
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs * ws, 1, 2).to(self.device)

        # generate anchor_wh tensor
        anchor_wh = self.anchor_size.repeat(hs * ws, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell, self.all_anchor_wh = self.create_grid(input_size)

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
        xywh_pred = torch.cat([xy_pred, wh_pred], -1).view(B, HW * ab_n, 4) * self.stride

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

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算交集的左上角点和右下角点的坐标
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算交集的宽高
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            # 计算交集的面积
            inter = w * h

            # 计算交并比
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 滤除超过nms阈值的检测框
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds

    def forward(self, x, target=None):
        # backbone主干网络
        _, c4, c5 = self.backbone(x)

        # head
        p5 = self.convsets_1(c5)

        # 处理c4特征
        p4 = self.reorg(self.route_layer(c4))

        # 融合
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        out = prediction
        if self.trainable:
            res = self.loss_func(out, target)
            return res.get('conf_loss'), res.get('cls_prob_loss'), res.get('position_loss'), res.get('iou_loss')
        else:
            B, abC, H, W = prediction.size()

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
            with torch.no_grad():
                # batch size = 1
                # 测试时，笔者默认batch是1，
                # 因此，我们不需要用batch这个维度，用[0]将其取走。
                # [B, H*W*num_anchor, 1] -> [H*W*num_anchor, 1]
                conf_pred = torch.sigmoid(conf_pred)[0]
                # [B, H*W*num_anchor, 4] -> [H*W*num_anchor, 4]
                bboxes = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)
                # [B, H*W*num_anchor, C] -> [H*W*num_anchor, C],
                scores = torch.softmax(cls_pred[0, :, :], dim=1) * conf_pred

                # 将预测放在cpu处理上，以便进行后处理
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # 后处理
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds


if __name__ == '__main__':
    GPU_ID = 1

    trainer_opt = YOLOV2TrainerConfig()
    data_opt = YOLOV2DataSetConfig()

    trainer_opt.device = 'cuda:{}'.format(GPU_ID)
    trainer_opt.lr = 1e-4

    net = YOLOv2(
        device=trainer_opt.device,
        input_size=416,
        trainable=True,
        anchor_size=YOLOV2DataSetConfig.pre_anchor_w_h,
        conf_thresh=YOLOV2TrainerConfig.score_th,
        nms_thresh=YOLOV2TrainerConfig.iou_th
    ).to(trainer_opt.device)

    # mean = [0.406, 0.456, 0.485]
    # std = [0.225, 0.224, 0.229]
    #
    # voc_train_loader = get_voc_data_loader(
    #     data_opt.root_path,
    #     ['2012'],
    #     data_opt.image_size,
    #     trainer_opt.batch_size,
    #     train=True,
    #     mean=mean,
    #     std=std
    # )
    #
    # voc_test_loader = get_voc_data_loader(
    #     data_opt.root_path,
    #     ['2012'],
    #     data_opt.image_size,
    #     trainer_opt.batch_size,
    #     train=False,
    #     mean=mean,
    #     std=std
    # )
    from yolo_v2_demo.utils.data import VOCDetection, BaseTransform, detection_collate
    from yolo_v2_demo.utils.augmentations import SSDAugmentation
    from yolo_v2_demo.utils.vocapi_evaluator import VOCAPIEvaluator
    from yolo_v2_demo.utils.data import BaseTransform

    ds = VOCDetection(
        root=data_opt.root_path,
        transform=SSDAugmentation(416)
    )
    dl = torch.utils.data.DataLoader(
                    ds,
                    batch_size=32,
                    shuffle=True,
                    collate_fn=detection_collate,
    )
    right_evaluator = VOCAPIEvaluator(
        data_opt.root_path,
        416,
        trainer_opt.device,
        transform=BaseTransform(416),
        labelmap=data_opt.kinds_name
    )

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=trainer_opt.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # net.eval()
    # net.trainable = False
    # right_evaluator.evaluate(net)
    for epoch in range(200):

        net.trainable = True
        net.set_grid(416)
        net.train()
        for iter_i, (images, targets) in enumerate(dl):
            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=416,
                                       stride=net.stride,
                                       label_lists=targets,
                                       anchor_size=net.anchor_size
                                       )
            # to device
            images = images.to(trainer_opt.device)
            targets = torch.tensor(targets).float().to(trainer_opt.device)

            conf_loss, cls_loss, bbox_loss, iou_loss = net(images, targets)
            loss = conf_loss + cls_loss + bbox_loss  # type: torch.Tensor

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_i % 20 == 0:
                print(
                    '{}'.format(epoch).center(5, ' ') + '{}'.format(iter_i).center(5, ' ') + 'loss:{:.5}'.format(loss.item()).center(10, ' ')
                )

        net.trainable = False
        net.set_grid(416)
        net.eval()
        right_evaluator.evaluate(net)







