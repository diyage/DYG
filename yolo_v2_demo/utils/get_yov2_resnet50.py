import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet50


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride

        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class YOLOV2Net(nn.Module):
    def __init__(
            self,
            pretrained: bool = True,
            num_anchors: int = 5,
            num_classes: int = 20,

    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = resnet50(pretrained)  # backbone
        # 检测头
        self.convsets_1 = nn.Sequential(
            Conv(2048, 1024, k=1),
            Conv(1024, 1024, k=3, p=1),
            Conv(1024, 1024, k=3, p=1)
        )

        # 融合高分辨率的特征信息
        self.route_layer = Conv(1024, 128, k=1)
        self.reorg = reorg_layer(stride=2)

        # 检测头
        self.convsets_2 = Conv(1024 + 128 * 4, 1024, k=3, p=1)

        # 预测曾
        self.pred = nn.Conv2d(1024, self.num_anchors * (1 + 4 + self.num_classes), 1)

    def forward(self, x: torch.Tensor):
        # backbone主干网络
        _, c4, c5 = self.backbone(x)

        # head
        p5 = self.convsets_1(c5)

        # 处理c4特征
        p4 = self.reorg(self.route_layer(c4))

        # 融合
        p5 = torch.cat([p4, p5], dim=1)

        # head
        p5 = self.convsets_2(p5)

        # 预测
        prediction = self.pred(p5)

        return prediction



