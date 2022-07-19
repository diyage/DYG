import torch
import torch.nn as nn


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self,  num_classes=1000):
        # https://zhuanlan.zhihu.com/p/105278156
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2 ,2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1),
            nn.MaxPool2d((2 ,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            nn.MaxPool2d((2 ,2), 2)
        )

        # output : stride = 16, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            nn.MaxPool2d((2 ,2), 2)
        )

        # output : stride = 32, c = 512
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            nn.MaxPool2d((2, 2), 2)
        )

        # output : stride = 32, c = 1024
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )

        self.conv_7 = nn.Conv2d(1024, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_extractor_top = None
        self.feature_extractor_down = None
        self.feature_extractor = None
        self.classifier = None
        self.translate_to_my_net()

    def translate_to_my_net(
            self
    ):
        tmp = [* self.conv_5]
        self.feature_extractor_top = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            *tmp[:-1],
        )
        self.feature_extractor_down = nn.Sequential(
            tmp[-1],
            self.conv_6,
        )
        self.feature_extractor = nn.Sequential(
            self.feature_extractor_top,
            self.feature_extractor_down,
        )
        self.classifier = nn.Sequential(
            self.conv_7,
            self.avgpool
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = self.conv_7(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def get_pretained_dark_net_19(
        path: str = '/home/dell/PycharmProjects/YOLO/pre_trained/darknet19_72.96.pth'
)->DarkNet_19:
    print('init pre-trained dark net 19'.center(50, '*'))
    saved_state_dict = torch.load(path)
    m = DarkNet_19()
    m.load_state_dict(saved_state_dict, strict=False)
    print('init successfully!'.center(50, '*'))
    return m

