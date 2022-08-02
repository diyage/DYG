import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .Predictor import YOLOV3Predictor
from .Tools import YOLOV3Tools
from Tool.BaseTools import BaseEvaluator


class YOLOV3Evaluator(BaseEvaluator):
    def __init__(
            self,
            model: nn.Module,
            predictor: YOLOV3Predictor,
            iou_th_for_make_target: float
    ):
        super().__init__()
        self.detector = model  # type: nn.Module
        self.backbone = model.backbone  # type: nn.Module
        self.device = next(model.parameters()).device

        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th_for_predict = self.predictor.iou_th
        self.anchor_keys = self.predictor.anchor_keys
        self.iou_th_for_make_target = iou_th_for_make_target

    def make_targets(
            self,
            labels
    ):
        targets = YOLOV3Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets

    def eval_detector_mAP(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval detector mAP',
    ):
        # compute mAP
        record = {
            key: [[], [], 0] for key in self.kinds_name
            # kind_name: [tp_list, score_list, gt_num]
        }
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.detector.eval()
            images = images.to(self.device)

            targets = self.make_targets(labels)
            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets, batch_size=images.shape[0])
            pre_decode = self.predictor.decode_predict(output, batch_size=images.shape[0])

            for image_index in range(images.shape[0]):

                res = YOLOV3Tools.get_pre_kind_name_tp_score_and_gt_num(
                    pre_decode[image_index],
                    gt_decode[image_index],
                    kinds_name=self.kinds_name,
                    iou_th=self.iou_th_for_predict
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
            recall, precision = YOLOV3Tools.calculate_pr(gt_num, tp_list, score_list)
            kind_name_ap = YOLOV3Tools.voc_ap(recall, precision)
            ap_vec.append(kind_name_ap)

        mAP = np.mean(ap_vec)
        print('\nmAP:{:.2%}'.format(mAP))
