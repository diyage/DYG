import torch
from Tool.FCNDesNet101_SS.Tools import SSTools
import numpy as np
import torch.nn.functional as F


class SSPredictor:
    def __init__(
            self,
    ):
        pass

    def decode_target(
            self,
            target: torch.Tensor,
    ) -> np.ndarray:
        target = SSTools.split_target(target)
        return target.cpu().detach().numpy()

    def decode_predict(
            self,
            predict: torch.Tensor,
    ) -> np.ndarray:
        predict = SSTools.split_predict(predict)
        pre_mask_vec = F.one_hot(predict.argmax(dim=-1), num_classes=predict.shape[-1])
        return pre_mask_vec.cpu().detach().numpy()
