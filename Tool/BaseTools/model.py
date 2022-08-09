'''
This packet is not important.
And its function equals to --make_predict-- (please see tools.BaseTools).
'''
from abc import abstractmethod
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(
            self,
            backbone: nn.Module
    ):
        super().__init__()
        self.backbone = backbone

    @abstractmethod
    def forward(
            self,
            *args,
            **kwargs
    ):
        '''

        Returns:

        '''
        pass
