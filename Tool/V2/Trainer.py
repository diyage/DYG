import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
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
    ):
        super().__init__(
            model,
            pre_anchor_w_h,
            image_size,
            grid_number,
            kinds_name
        )

        self.backbone = model.backbone  # type: nn.Module
        self.backbone.cuda()
        # be careful, darknet19 is not the detector

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ):
        return YOLOV2Tools.make_targets(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            need_abs
        )


