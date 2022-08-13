from .Tools import YOLOV4Tools
from .Model import YOLOV4Model
from Tool.BaseTools import BaseTrainer
from typing import Union


class YOLOV4Trainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOV4Model,
            pre_anchor_w_h_rate: dict,
            image_size: tuple,
            image_shrink_rate: dict,
            kinds_name: list,
            iou_th_for_make_target: float,
            multi_gt: bool,
    ):
        super().__init__(
            model,
            pre_anchor_w_h_rate,
            image_size,
            image_shrink_rate,
            kinds_name,
            iou_th_for_make_target,
        )

        self.anchor_keys = list(pre_anchor_w_h_rate.keys())
        self.multi_gt = multi_gt

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV4Tools.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels,
    ):
        targets = YOLOV4Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
            multi_gt=self.multi_gt
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets



