import torch.nn as nn
from .Tools import YOLOV1Tools
from Tool.BaseTools import BaseTrainer


class YOLOV1Trainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            pre_anchor_w_h: tuple,
            image_size: tuple,
            grid_number: tuple,
            kinds_name: list,
    ):
        super().__init__(
            model,
            pre_anchor_w_h,
            image_size,
            grid_number,
            kinds_name
        )

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ):
        return YOLOV1Tools.make_targets(
            labels,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            need_abs
        )

