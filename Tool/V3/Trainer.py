import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .Tools import YOLOV3Tools
from Tool.BaseTools import BaseTrainer
from typing import Union


class YOLOV3Trainer(BaseTrainer):
    def __init__(
            self,
            model: nn.Module,
            pre_anchor_w_h: Union[tuple, dict],
            image_size: tuple,
            grid_number: Union[tuple, dict],
            kinds_name: list,
            iou_th: float = 0.6
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
            self.iou_th,
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets



