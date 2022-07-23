import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo_v2_demo.utils.get_yov2_resnet50 import Conv, reorg_layer
from yolo_v2_demo.utils.backbone import *
import numpy as np
import tools
from Tool.V2 import *
from Tool.BaseTools import get_voc_data_loader


class YOLOv2(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.6,
                 anchor_size=None):
        super(YOLOv2, self).__init__()
        self.device = device
        self.input_size = input_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
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

        B, abC, H, W = prediction.size()
        target = target.view(B, H*W*self.num_anchors, -1)

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

        # train
        if self.trainable:
            txtytwth_pred = txtytwth_pred.view(B, H * W, self.num_anchors, 4)
            # decode bbox
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, 7:].view(-1, 4)

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
            conf_loss, cls_loss, bbox_loss, iou_loss = tools.loss(pred_conf=conf_pred,
                                                                  pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,
                                                                  pred_iou=iou_pred,
                                                                  label=target,
                                                                  num_classes=self.num_classes
                                                                  )
            return {
                'total_loss': conf_loss + cls_loss + bbox_loss,
                'position_loss': bbox_loss,
                'conf_loss': conf_loss,
                'cls_prob_loss': cls_loss,
                'iou_loss': iou_loss,
            }, out
        else:

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

                return bboxes, scores, cls_inds, out


class MyEvaluator(YOLOV2Evaluator):
    def __init__(
            self,
            model: YOLOv2,
            predictor: YOLOV2Predictor
    ):
        super().__init__(
            model,
            predictor
        )
        self.detector = model  # type: nn.Module
        self.backbone = model.backbone  # type: nn.Module
        self.device = next(model.parameters()).device

        # be careful, darknet19 is not the detector
        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th = self.predictor.iou_th

    def eval_detector_mAP(
            self,
            data_loader_test,
            desc: str = 'eval detector mAP',
    ):
        # compute mAP
        record = {
            key: [[], [], 0] for key in self.kinds_name
            # kind_name: [tp_list, score_list, gt_num]
        }
        for batch_id, (images, labels) in enumerate(data_loader_test):
            self.detector.eval()
            self.detector.trainable = False

            images = images.to(self.device)
            targets = self.make_targets(labels).to(self.device)

            output_vec = []

            for image_index in range(images.shape[0]):
                bboxes, scores, cls_inds, output = self.detector(
                    images[image_index].unsqueeze(dim=0),
                    targets[image_index].unsqueeze(dim=0)
                )
                output_vec.append(output)

            output = torch.cat(output_vec, dim=0).to(images.device)

            gt_decode = self.predictor.decode(targets, out_is_target=True)
            pre_decode = self.predictor.decode(output, out_is_target=False)

            for image_index in range(images.shape[0]):

                res = YOLOV2Tools.get_pre_kind_name_tp_score_and_gt_num(
                    pre_decode[image_index],
                    gt_decode[image_index],
                    kinds_name=self.kinds_name,
                    iou_th=self.iou_th
                )

                for pre_kind_name, is_tp, pre_score in res[0]:
                    record[pre_kind_name][0].append(is_tp)  # tp list
                    record[pre_kind_name][1].append(pre_score)  # score list

                for kind_name, gt_num in res[1].items():
                    record[kind_name][2] += gt_num

        # end for dataloader
        ap_vec = []
        for kind_name in self.kinds_name:
            tp_list, score_list, gt_num = record[kind_name]
            recall, precision = YOLOV2Tools.calculate_pr(gt_num, tp_list, score_list)
            kind_name_ap = YOLOV2Tools.voc_ap(recall, precision)
            ap_vec.append(kind_name_ap)

        mAP = np.mean(ap_vec)
        print('\nmAP:{:.2%}'.format(mAP))


if __name__ == '__main__':
    GPU_ID = 1
    LOSS_TYPE = 1
    YOLOV2Predictor.TYPE = LOSS_TYPE
    YOLOV2Tools.TYPE = LOSS_TYPE
    fast_load = False

    trainer_opt = YOLOV2TrainerConfig()
    data_opt = YOLOV2DataSetConfig()

    trainer_opt.device = 'cuda:{}'.format(GPU_ID)
    trainer_opt.lr = 1e-4

    net = YOLOv2(
        device=trainer_opt.device,
        input_size=416,
        trainable=False,
        anchor_size=YOLOV2DataSetConfig.pre_anchor_w_h,
        conf_thresh=YOLOV2TrainerConfig.score_th,
        nms_thresh=YOLOV2TrainerConfig.iou_th
    )
    net.to(trainer_opt.device)

    voc_train_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=True,
    )
    voc_test_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=False,
    )
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=1e-4,
        momentum=0.9,
        weight_decay=5e-4
    )

    my_predictor = YOLOV2Predictor(
        trainer_opt.iou_th,
        trainer_opt.prob_th,
        trainer_opt.conf_th,
        trainer_opt.score_th,
        data_opt.pre_anchor_w_h,
        data_opt.kinds_name,
        data_opt.image_size,
        data_opt.grid_number
    )
    my_evaluator = MyEvaluator(
        net,
        my_predictor
    )
    for epoch in range(200):

        net.train()
        net.trainable = True
        for batch_id, (images, labels) in enumerate(voc_train_loader):
            targets = YOLOV2Tools.make_targets(
                labels,
                data_opt.pre_anchor_w_h,
                data_opt.image_size,
                data_opt.grid_number,
                data_opt.kinds_name,
                trainer_opt.iou_th,
            )
            images = images.to(trainer_opt.device)
            targets = targets.to(trainer_opt.device)
            loss_dict, out = net(images, targets)
            loss = loss_dict.get('total_loss')  # type: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                '{}'.format(epoch).center(5, ' ') + '{}'.format(batch_id).center(5, ' ') + 'loss:{:.5}'.format(loss.item()).center(10, ' ')
            )

        net.eval()
        net.trainable = False
        my_evaluator.eval_detector_mAP(voc_test_loader)







