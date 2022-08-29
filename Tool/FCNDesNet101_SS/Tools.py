from Tool.BaseTools import BaseTools
import torch
import numpy as np


class SSTools(BaseTools):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_grid_number_and_pre_anchor_w_h(
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def make_target(
            labels: list,
            *args,
            **kwargs
    ) -> torch.Tensor:
        # masks_vec = labels
        return torch.from_numpy(np.array(labels))

    @staticmethod
    def split_target(
            target: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        return target.permute(0, 2, 3, 1)

    @staticmethod
    def split_predict(
            out_put: torch.Tensor,
            *args,
            **kwargs
    ) -> torch.Tensor:
        return out_put.permute(0, 2, 3, 1)
