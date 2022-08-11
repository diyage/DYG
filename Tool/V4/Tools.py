import torch
import numpy as np
from Tool.BaseTools import BaseTools


class YOLOV4Tools(BaseTools):
    @staticmethod
    def make_target(
            *args,
            **kwargs
    ):
        pass

    @staticmethod
    def split_target(
            *args,
            **kwargs
    ) -> dict:
        pass

    @staticmethod
    def split_predict(
            *args,
            **kwargs
    ) -> dict:
        pass

    @staticmethod
    def mish(
            x: torch.Tensor
    ) -> torch.Tensor:
        return x*torch.tanh(x*torch.log(1+torch.exp(x)))
    
