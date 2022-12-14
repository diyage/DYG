import torch.nn as nn
from .Predictor import YOLOV2Predictor
from .Tools import YOLOV2Tools
from .Model import YOLOV2Model
from Tool.BaseTools import BaseEvaluator


class YOLOV2Evaluator(BaseEvaluator):
    def __init__(
            self,
            model: YOLOV2Model,
            predictor: YOLOV2Predictor,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target
        )

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV2Tools.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
    ):
        return YOLOV2Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            iou_th=self.iou_th_for_make_target,
        ).to(self.device)
