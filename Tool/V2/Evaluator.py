import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from .Predictor import Predictor
from .BaseTools import YOLOV2Tools


class Evaluator:
    def __init__(
            self,
            model: nn.Module,
            predictor: Predictor
    ):
        self.detector = model  # type: nn.Module
        self.detector.cuda()

        self.dark_net = model.darknet19  # type: nn.Module
        self.dark_net.cuda()
        # be careful, darknet19 is not the detector
        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th = self.predictor.iou_th

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ):
        return YOLOV2Tools.make_targets(labels,
                                        self.pre_anchor_w_h,
                                        self.image_size,
                                        self.grid_number,
                                        self.kinds_name,
                                        need_abs)

    def eval_classifier(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval classifier'
    ):
        vec = []
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.dark_net.eval()
            images = images.cuda()  # type: torch.Tensor
            labels = labels.cuda()  # type: torch.Tensor

            output = self.dark_net(images)  # type: torch.Tensor
            acc = (output.argmax(dim=-1) == labels).float().mean()
            vec.append(acc)

        accuracy = sum(vec) / len(vec)
        print('Acc: {:.2%}'.format(accuracy.item()))

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
            images = images.cuda()

            targets = self.make_targets(labels, need_abs=True).cuda()
            output = self.detector(images)

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
