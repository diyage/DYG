'''
This packet(predictor) is core of Object Detection.
It will be used in inference phase for decoding ground-truth(target)/model-output(predict).

You must know some special definitions in my frame:
    kps_vec_s --> [kps_vec0, kps_vec1, ...]                         for batch images
        kps_vec --> [kps0, kps1, kps2, ...]                         for one image
            kps --> (predict_kind_name, abs_double_pos, score)      for one object
                predict_kind_name --> str  (e.g. 'cat', 'dog', 'car', ...)
                abs_double_pos --> (x, y, x, y)   scaled on image
                score --> float   conf * cls_prob
'''
import torch
from Tool.BaseTools.tools import BaseTools
from typing import Union
from abc import abstractmethod
from typing import List


class BasePredictor:
    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            score_th: float,
            pre_anchor_w_h_rate: Union[tuple, dict],
            kinds_name: list,
            image_size: tuple,
            image_shrink_rate: Union[tuple, dict]
    ):
        self.iou_th = iou_th
        self.prob_th = prob_th
        self.conf_th = conf_th
        self.score_th = score_th
        self.kinds_name = kinds_name

        self.pre_anchor_w_h_rate = pre_anchor_w_h_rate
        self.pre_anchor_w_h = None

        self.image_shrink_rate = image_shrink_rate
        self.grid_number = None

        self.image_size = None
        self.change_image_wh(image_size)

    @abstractmethod
    def change_image_wh(
            self,
            image_wh: tuple
    ):
        pass

    def nms(
            self,
            position_abs_: torch.Tensor,
            scores_max_value: torch.Tensor,
            scores_max_index: torch.Tensor
    ) -> List:
        '''
        for one image, all predicted objects( already mask some bad ones)
        it may have many kinds...
        Args:
            position_abs_: (P, 4)
            scores_max_value: (P, ) predicted kind_name's score
            scores_max_index: (P, ) predicted kind_name's index

        Returns: kps_vec
        '''
        def for_response(
                now_kind_pos_abs,
                now_kind_scores_max_value,
        ):
            res = []
            keep_index = BaseTools.nms(
                now_kind_pos_abs,
                now_kind_scores_max_value,
                threshold=iou_th,
            )

            for index in keep_index:
                s = now_kind_scores_max_value[index]
                abs_double_pos = tuple(now_kind_pos_abs[index].cpu().detach().numpy().tolist())
                predict_kind_name = kind_name

                res.append(
                    (predict_kind_name, abs_double_pos, s.item())  # kps
                )

            return res

        iou_th = self.iou_th
        kinds_name = self.kinds_name

        total = []
        for kind_index, kind_name in enumerate(kinds_name):
            now_kind_response = scores_max_index == kind_index
            total = total + for_response(
                position_abs_[now_kind_response],
                scores_max_value[now_kind_response],
            )

        return total

    @abstractmethod
    def decode_one_target(
            self,
            *args,
            **kwargs
    ) -> List:
        '''
        Returns:
            kps_vec
        '''
        pass

    @abstractmethod
    def decode_target(
            self,
            *args,
            **kwargs
    ) -> List[List]:
        '''

                Returns:
                    kps_vec_s
        '''
        pass

    @abstractmethod
    def decode_one_predict(
            self,
            *args,
            **kwargs
    ) -> List:
        '''

                        Returns:
                            kps_vec
                '''
        pass

    @abstractmethod
    def decode_predict(
            self,
            *args,
            **kwargs
    ) -> List[List]:
        '''

                        Returns:
                            kps_vec_s
                '''
        pass

