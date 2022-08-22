from Tool.V4.Model import YOLOV4Model, CSPDarkNet53, CBL
import torch
import torch.nn as nn
from typing import *


class CSPDarkNet53IS(CSPDarkNet53):
    def __init__(
            self,
    ):
        super().__init__()

    def forward(
            self,
            x: torch.Tensor
    ):
        outputs = []
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["act1"](x)

        for i, layer_name in enumerate(self.layer_names):
            layer = self.backbone[layer_name]
            x = layer(x)
            outputs.append(x)
        return outputs  # C1, C2, C3, C4, C5


class Process6(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.cbl_s_0 = nn.Sequential(
            CBL(128, 64, 1, 1, 0),

        )
        self.up_sample_1 = nn.Sequential(
            CBL(256, 64, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(
            self,
            c2: torch.Tensor,
            y76: torch.Tensor
    ):
        tmp0 = self.cbl_s_0(c2)  # (-1, 64, 152, 152)
        tmp1 = self.up_sample_1(y76)  # (-1, 64, 152, 152)
        y152 = torch.cat((tmp0, tmp1), dim=1)  # (-1, 128, 152, 152)
        return y152  # (-1, 128, 152, 152)


class Process7(nn.Module):
    def __init__(
            self,
    ):
        super().__init__()
        self.cbl_s_0 = nn.Sequential(
            CBL(64, 32, 1, 1, 0),

        )
        self.up_sample_1 = nn.Sequential(
            CBL(128, 32, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(
            self,
            c1: torch.Tensor,
            y152: torch.Tensor
    ):
        tmp0 = self.cbl_s_0(c1)  # (-1, 32, 304, 304)
        tmp1 = self.up_sample_1(y152)  # (-1, 32, 304, 304)
        y304 = torch.cat((tmp0, tmp1), dim=1)  # (-1, 64, 304, 304)
        return y304  # (-1, 64, 304, 304)


class SemanticSegmentationNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.process6 = Process6()
        self.process7 = Process7()

    def forward(
            self,
            c1: torch.Tensor,
            c2: torch.Tensor,
            y76: torch.Tensor
    ):
        y152 = self.process6(c2, y76)
        y304 = self.process7(c1, y152)
        return y304  # (-1, 64, 304, 304)


class SemanticSegmentationHead(nn.Module):
    def __init__(
            self,
            num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.head = nn.Sequential(

            CBL(64, 32, 1, 1, 0),  # y76 (-1, 64, 304, 304)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBL(32, 64, 3, 1, 1),
            CBL(64, 32, 1, 1, 0),
            CBL(32, 64, 3, 1, 1),
            CBL(64, 32, 1, 1, 0),
            CBL(32, 64, 3, 1, 1),
            nn.Conv2d(64, num_classes + 1, 1, 1, 0),
            nn.Sigmoid(),
        )

    def forward(
            self,
            y304: torch.Tensor
    ):
        mask = self.head(y304)  # (-1, num_cls + 1, 608, 608)
        return mask


class YOLOV4ForISModel(YOLOV4Model):
    def __init__(
            self,
            backbone: Union[nn.Module, CSPDarkNet53, CSPDarkNet53IS],
            num_anchors_for_single_size: int = 3,
            num_classes: int = 20,
    ):
        super().__init__(
            backbone,
            num_anchors_for_single_size,
            num_classes
        )
        self.mask_neck = SemanticSegmentationNeck()
        self.mask_head = SemanticSegmentationHead(num_classes)

    def forward(
            self,
            x: torch.Tensor
    ):
        c1, c2, c3, c4, c5 = self.backbone(x)
        y76, y38_a, y19_a = self.neck(c3, c4, c5)

        y304 = self.mask_neck(c1, c2, y76)

        y76_o, y38_o, y19_o = self.head(y76, y38_a, y19_a)
        mask = self.mask_head(y304)

        return {
            'for_s': y76_o,  # s=8
            'for_m': y38_o,  # s=16
            'for_l': y19_o,  # s=32
            'mask': mask
        }


def debug_CSPDarkNet53IS():
    m = CSPDarkNet53IS()
    x = torch.rand(size=(1, 3, 608, 608))
    out = m(x)
    for val in out:
        print(val.shape)


def debug_YOLOV4ISModel():
    x = torch.rand(size=(1, 3, 608, 608))
    backbone = CSPDarkNet53IS()
    net = YOLOV4ForISModel(backbone, 3, 20)
    out = net(x)
    for key, val in out.items():
        print('{}:{}'.format(key, val.shape))


if __name__ == '__main__':
    # debug_CSPDarkNet53IS()
    debug_YOLOV4ISModel()