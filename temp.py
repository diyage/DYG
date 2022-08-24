import random
import numpy as np
res = {
    'x': [[900, 3], [900, 3], ...],
    'y': [[300, 3], [300, 3], ...],
}

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(
            self,
            in_feture: int = 900,
            out_feature: int = 300,
            dims: int = 3,
    ):
        super().__init__()
        self.in_feture = in_feture
        self.out_feature = out_feature
        self.dims = dims
        self.model = nn.Sequential(
            nn.Linear(dims * in_feture, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, dims * out_feature),
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        assert len(x.shape) == 3
        x = x.view(x.shape[0], -1)

        out = self.model(x)  # (-1, dim*out_feature)
        return out.view(-1, self.out_feature, self.dims)


x = torch.rand(size=(64, 900, 3))
net = Net()
pre = net(x)
print(pre.shape)