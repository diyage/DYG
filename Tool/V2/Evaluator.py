import torch.nn as nn
from .Predictor import YOLOV2Predictor
from .Tools import YOLOV2Tools
from Tool.BaseTools import BaseEvaluator


class YOLOV2Evaluator(BaseEvaluator):
    def __init__(
            self,
            model: nn.Module,
            predictor: YOLOV2Predictor
    ):
        super().__init__(
            detector=model,
            device=next(model.parameters()).device,
            predictor=predictor,
            kinds_name=predictor.kinds_name,
            iou_th=predictor.iou_th
        )

        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number

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
            iou_th=self.iou_th,
        ).to(self.device)
