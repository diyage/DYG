import torch.nn as nn
from .Tools import YOLOV2Tools
from .Model import YOLOV2Model
from Tool.BaseTools import BaseTrainer


class YOLOV2Trainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOV2Model,
            pre_anchor_w_h_rate: tuple,
            image_size: tuple,
            image_shrink_rate: tuple,
            kinds_name: list,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            pre_anchor_w_h_rate,
            image_size,
            image_shrink_rate,
            kinds_name,
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


