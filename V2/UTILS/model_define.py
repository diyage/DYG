import torch
import torch.nn as nn


def make_conv_bn_active_layer(
        in_channel: int,
        out_channel: int,
        kernel_size: tuple = (3, 3),
        stride: tuple = (1, 1),
        padding: tuple = (1, 1),
        active: nn.Module = nn.LeakyReLU):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channel),
        active(),
    )


class DarkNet19(nn.Module):
    # https://zhuanlan.zhihu.com/p/35325884
    def __init__(self,
                 kinds_number: int = 1000):
        super().__init__()
        self.feature_extractor_top = nn.Sequential(
            make_conv_bn_active_layer(3, 32),
            nn.MaxPool2d(kernel_size=(2, 2)),

            make_conv_bn_active_layer(32, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),

            make_conv_bn_active_layer(64, 128),
            make_conv_bn_active_layer(128, 64, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(64, 128),
            nn.MaxPool2d(kernel_size=(2, 2)),

            make_conv_bn_active_layer(128, 256),
            make_conv_bn_active_layer(256, 128, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(128, 256),
            nn.MaxPool2d(kernel_size=(2, 2)),

            make_conv_bn_active_layer(256, 512),
            make_conv_bn_active_layer(512, 256, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(256, 512),
            make_conv_bn_active_layer(512, 256, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(256, 512),
        )
        self.feature_extractor_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2)),
            make_conv_bn_active_layer(512, 1024),
            make_conv_bn_active_layer(1024, 512, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(512, 1024),
            make_conv_bn_active_layer(1024, 512, kernel_size=(1, 1), padding=(0, 0)),
            make_conv_bn_active_layer(512, 1024),
        )
        self.feature_extractor = nn.Sequential(
            self.feature_extractor_top,
            self.feature_extractor_down,
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, kinds_number, kernel_size=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

    def forward(self, x: torch.Tensor):
        out = self.feature_extractor(x)
        out = self.classifier(out)
        return out.view(out.shape[0], -1)


class YOLOV2Net(nn.Module):
    def __init__(self,
                 darknet19: DarkNet19):
        super().__init__()
        self.darknet19 = darknet19

        self.pass_through_conv = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=(1, 1))
        )

        self.conv3_1024 = nn.Sequential(
            make_conv_bn_active_layer(1024, 1024),
            make_conv_bn_active_layer(1024, 1024),
            make_conv_bn_active_layer(1024, 1024),
        )

        self.conv3_1_out = nn.Sequential(
            make_conv_bn_active_layer(1280, 1024),
            nn.Conv2d(1024, 125, kernel_size=(1, 1))
        )

    def pass_through(self, a: torch.Tensor):
        # https://zhuanlan.zhihu.com/p/35325884
        N = a.shape[0]
        assert a.shape == (N, 512, 26, 26)
        tmp = self.pass_through_conv(a)  # N * 64 * 26 * 26
        N, C, H, W = tmp.shape
        tmp = nn.Unfold(kernel_size=2, stride=2)(tmp) # N * (C * 4) * (H/2 * W/2)
        tmp = tmp.view(N, C*4, H//2, W//2)  # N * 256 * 13 * 13
        return tmp

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        assert x.shape == (N, 3, 416, 416)

        a = self.darknet19.feature_extractor_top(x)  # N * 512 * 26 * 26
        a_ = self.pass_through(a)  # N * 256 * 13 * 13

        b = self.darknet19.feature_extractor_down(a)  # N * 1024 * 13 * 13
        b_ = self.conv3_1024(b)  # N * 1024 * 13 * 13

        d = torch.cat((b_, a_), dim=1)  # N * (1024 + 256) * 13 * 13

        return self.conv3_1_out(d)  # N * 125 * 13 * 13

