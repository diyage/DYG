import torch.nn as nn
from .Predictor import YOLOV3Predictor
from .Tools import YOLOV3Tools
from Tool.BaseTools import BaseEvaluator


class YOLOV3Evaluator(BaseEvaluator):
    def __init__(
            self,
            model: nn.Module,
            device: str,
            predictor: YOLOV3Predictor,
            iou_th_for_make_target: float
    ):
        super().__init__(
            detector=model,
            device=device,
            predictor=predictor,
            kinds_name=predictor.kinds_name,
            iou_th=predictor.iou_th
        )

        self.predictor = predictor

        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
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
