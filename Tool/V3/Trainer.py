from .Model import YOLOV3Model
from .Tools import YOLOV3Tools
from Tool.BaseTools import BaseTrainer
from typing import Union


class YOLOV3Trainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOV3Model,
            pre_anchor_w_h: Union[tuple, dict],
            image_size: tuple,
            grid_number: Union[tuple, dict],
            kinds_name: list,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            pre_anchor_w_h,
            image_size,
            grid_number,
            kinds_name,
            iou_th_for_make_target,
        )

        self.anchor_keys = list(pre_anchor_w_h.keys())

    def make_targets(
            self,
            labels,
    ):
        targets = YOLOV3Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets



