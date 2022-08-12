import torch.nn as nn
from .Tools import YOLOV2Tools
from .Model import YOLOV2Model
from Tool.BaseTools import BaseTrainer


class YOLOV2Trainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOV2Model,
            pre_anchor_w_h: tuple,
            image_size: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            pre_anchor_w_h,
            image_size,
            grid_number,
            kinds_name,
            iou_th_for_make_target
        )

    def make_targets(
            self,
            labels,
    ):
        return YOLOV2Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
        ).to(self.device)


