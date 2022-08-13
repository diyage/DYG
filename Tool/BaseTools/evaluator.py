'''
Used for evaluating some metrics of detector.
'''
from abc import abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm
from .tools import BaseTools
from .predictor import BasePredictor
import numpy as np
from .model import BaseModel
import torch


class BaseEvaluator:
    def __init__(
            self,
            model: BaseModel,
            predictor: BasePredictor,
            iou_th_for_make_target: float
    ):
        self.detector = model  # type: BaseModel
        self.device = next(model.parameters()).device

        self.predictor = predictor

        self.kinds_name = predictor.kinds_name
        self.iou_th_for_eval = self.predictor.iou_th

        self.pre_anchor_w_h_rate = self.predictor.pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = self.predictor.image_shrink_rate
        self.grid_number = None

        self.image_size = None  # type: tuple
        self.change_image_wh(self.predictor.image_size)

        self.iou_th_for_make_target = iou_th_for_make_target

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    def __no_grad_eval(
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

            gt_decode = self.predictor.decode_target(targets)  # kps_vec_s
            pre_decode = self.predictor.decode_predict(output)  # kps_vec_s

            for image_index in range(images.shape[0]):

                res = BaseTools.get_pre_kind_name_tp_score_and_gt_num(
                    pre_decode[image_index],
                    gt_decode[image_index],
                    kinds_name=self.kinds_name,
                    iou_th=self.iou_th_for_eval
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
            recall, precision = BaseTools.calculate_pr(gt_num, tp_list, score_list)
            kind_name_ap = BaseTools.voc_ap(recall, precision)
            ap_vec.append(kind_name_ap)

        mAP = np.mean(ap_vec)
        print('\nmAP:{:.2%}'.format(mAP))

    def eval_detector_mAP(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval detector mAP',
    ):
        with torch.no_grad():
            self.__no_grad_eval(
                data_loader_test,
                desc
            )

