import torchvision.models as models
import torch
import torch.nn as nn


class FCNResnet101(nn.Module):
    def __init__(
            self,
            pretrained: bool = False,
            num_classes: int = 21,
    ):
        super().__init__()
        self.__model = models.segmentation.fcn_resnet101(
                                pretrained=pretrained,
                                num_classes=num_classes
                            )

    def forward(
            self,
            x: torch.Tensor
    ):
        res = self.__model(x)
        return res['out']


def get_fcn_resnet101(
        pretrained: bool = False,
        num_classes: int = 21,
) -> FCNResnet101:
    return FCNResnet101(
        pretrained,
        num_classes,
    )


def de_bug_model():
    x = torch.rand(size=(1, 3, 448, 448))
    m = get_fcn_resnet101(True, 21)
    y = m(x)
    # print(m)
    print(y.shape)
    print(y)


if __name__ == '__main__':
    de_bug_model()
