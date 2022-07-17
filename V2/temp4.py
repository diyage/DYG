import torch
import torch.nn as nn
import torch.nn.functional as F


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(
            self,
            stride=2):
        super(Reorg, self).__init__()

        self.stride = stride

    def forward(self, x):
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

        ws, hs = self.stride, self.stride
        x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x


def pass_through(
        x: torch.Tensor,
        stride: tuple = (2, 2)):

    assert (x.data.dim() == 4)
    B = x.data.size(0)
    C = x.data.size(1)
    H = x.data.size(2)
    W = x.data.size(3)

    ws, hs = stride
    x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
    x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
    x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
    x = x.view(B, hs * ws * C, H // hs, W // ws)
    return x


a = torch.rand(size=(1, 8, 64, 64))
m = Reorg()
b = m(a)
c = pass_through(a)
print(b == c)