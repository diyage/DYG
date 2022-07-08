import torch
import torch.nn as nn


class YoLoV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),  # 64 * 224 *224
            nn.LeakyReLU(),

            nn.MaxPool2d(2, 2),  # 64 * 112 *112

            nn.Conv2d(64, 192, 3, 1, 1),  # 192 * 112 *112
            nn.LeakyReLU(),

            nn.MaxPool2d(2, 2),  # 192 * 56 *56

            nn.Conv2d(192, 256, 1, 1, 0),  # 128 * 56 *56
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),  # 512 * 56 *56
            nn.LeakyReLU(),

            nn.MaxPool2d(2, 2),  # 512 * 28 * 28

            nn.Conv2d(512, 256, 1, 1, 0),  # 256 * 28 *28
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 3, 1, 1),  # 512 * 28 *28
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 1, 1, 0),  # 512 * 28 *28
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),  # 1024 * 28 *28
            nn.LeakyReLU(),

            nn.MaxPool2d(2, 2),  # 1024 * 14 * 14

            nn.Conv2d(1024, 512, 1, 1, 0),  # 512 * 14 *14
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 3, 1, 1),  # 1024 * 14 *14
            nn.LeakyReLU(),

            nn.Conv2d(1024, 1024, 3, 2, 1),  # 1024 * 7 *7
            nn.LeakyReLU(),

        )
        self.detector = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1470),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor)->torch.Tensor:
        x = self.feature_extractor(x)  # type: torch.Tensor
        x = x.view(x.shape[0], -1)
        x = self.detector(x)  # type: torch.Tensor
        x = x.view(x.shape[0], 7, 7, -1)
        return x


