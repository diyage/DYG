import torch.nn as nn
from .Tools import YOLOV2Tools
from Tool.BaseTools import BaseTrainer


class YOLOV2Trainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            pre_anchor_w_h: tuple,
            image_size: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th:float = 0.6
    ):
        super().__init__(
            model,
            pre_anchor_w_h,
            image_size,
            grid_number,
            kinds_name,
            iou_th
        )
        self.backbone = model.backbone  # type: nn.Module

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
            self.iou_th,
        ).to(self.device)


