from Tool.BaseTools.cv2_ import CV2
from Tool.BaseTools.tools import BaseTools
import numpy as np


class MyRandomFlip:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, labels: list):
        code = np.random.randint(-1, 5)
        H, W, C = image.shape
        if code >= 2:
            return image, labels
        elif code == -1:  # for x and y axis
            new_image = CV2.flip(image, -1)
            new_labels = []
            for obj in labels:
                pos = obj[1:]
                new_pos = (W - pos[2], H - pos[3], W - pos[0], H - pos[0])
                new_labels.append((obj[0], *new_pos))

            return new_image, new_labels
        elif code == 0:  # for y axis
            new_image = CV2.flip(image, 0)
            new_labels = []
            for obj in labels:
                pos = obj[1:]
                new_pos = (pos[0], H - pos[3], pos[2], H - pos[0])
                new_labels.append((obj[0], *new_pos))
            return new_image, new_labels
        else:  # code == 1 for x axis
            new_image = CV2.flip(image, 1)
            new_labels = []
            for obj in labels:
                pos = obj[1:]
                new_pos = (W - pos[2], pos[1], W - pos[0], pos[3])
                new_labels.append((obj[0], *new_pos))
            return new_image, new_labels


class MyRandomNoise:
    def __init__(
            self,
            rate: float = 0.1
    ):
        self.rate = rate

    def __call__(self, image: np.ndarray, labels: list):
        code = np.random.randint(0, 3)
        if code == 0:
            return image, labels
        elif code == 1:
            noise = np.random.uniform(0, self.rate*255.0, size=image.shape)
            new_image = np.clip(image + noise, 0, 255)
            return new_image.astype(np.uint8), labels
        else:
            noise = self.rate * np.random.randn(*image.shape)
            new_image = 1.0 * image/255.0 + noise
            new_image = 255.0 * np.clip(new_image, 0.0, 1.0)
            return new_image.astype(np.uint8), labels


class MyToTensor:
    def __init__(
            self,
            mean=None,
            std=None,
    ):
        self.mean = mean
        self.std = std
        if self.mean is None or self.std is None:
            self.mean = [0, 0, 0]
            self.std = [1, 1, 1]

    def __call__(self, image: np.ndarray, labels: list):
        # image is a BGR uint8 image
        image_tensor = BaseTools.image_np_to_tensor(
            image,
            self.mean,
            self.std
        )
        return image_tensor, labels


class MyRandomZeroOneChannel:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray, labels: list):
        code = np.random.randint(0, 30)
        # a lot of original images
        H, W, C = image.shape
        if code == 0:
            new_image = np.concatenate(
                (
                 np.zeros(shape=(H, W, 1)).astype(np.uint8),
                 image[..., 1:2],
                 image[..., 2:3]
                ),
                axis=2
            )
            return new_image, labels
        elif code == 1:
            new_image = np.concatenate(
                (
                 image[..., 0:1],
                 np.zeros(shape=(H, W, 1)).astype(np.uint8),
                 image[..., 2:3]
                ),
                axis=2
            )
            return new_image, labels
        elif code == 2:
            new_image = np.concatenate(
                (
                    image[..., 0:1],
                    image[..., 1:2],
                    np.zeros(shape=(H, W, 1)).astype(np.uint8),
                ),
                axis=2
            )
            return new_image, labels
        else:
            return image, labels


class MyRandomWeightedChannel:
    def __init__(
            self,
            min_: float = 0.5,
            max_: float = 1.5,
    ):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, image: np.ndarray, labels: list):

        # a lot of original images
        H, W, C = image.shape
        new_image = image
        for c in range(C):
            code = np.random.randint(0, 5)
            if code == 0:  # weighted
                if c == 0:
                    weight = np.random.uniform(self.min_, self.max_, size=(H, W, 1))
                    tmp = np.clip(weight * new_image[..., 0:1], 0, 255).astype(np.uint8)
                    new_image = np.concatenate(
                        (
                            tmp,
                            new_image[..., 1:2],
                            new_image[..., 2:3]
                        ),
                        axis=2
                    )
                elif c == 1:
                    weight = np.random.uniform(self.min_, self.max_, size=(H, W, 1))
                    tmp = np.clip(weight * new_image[..., 1:2], 0, 255).astype(np.uint8)
                    new_image = np.concatenate(
                        (
                            new_image[..., 0:1],
                            tmp,
                            new_image[..., 2:3]
                        ),
                        axis=2
                    )
                elif c == 2:
                    weight = np.random.uniform(self.min_, self.max_, size=(H, W, 1))
                    tmp = np.clip(weight * new_image[..., 2:3], 0, 255).astype(np.uint8)
                    new_image = np.concatenate(
                        (
                            new_image[..., 0:1],
                            new_image[..., 1:2],
                            tmp,
                        ),
                        axis=2
                    )
            else:  # keep origin
                continue
        return new_image, labels


class MyCompose:
    def __init__(
            self,
            all_trans: list = [],
    ):
        self.all_trans = all_trans

    def __call__(self, image: np.ndarray, labels: list):
        for trans in self.all_trans:
            image, labels = trans(image, labels)
        return image, labels


if __name__ == '__main__':
    img = CV2.imread(r'D:\Software\PyCharm\files\YOLO\2008_000200.jpg')
    print(img[..., 0:1].shape)
    l = [('dog', 20, 25, 120, 260)]

    R = MyCompose([
        MyRandomFlip(),
        MyRandomNoise(0.1),
        MyRandomWeightedChannel(0.5, 1.5),
        MyRandomZeroOneChannel(),
        # MyToTensor()
    ])

    new_img, new_l = R(img, l)
    CV2.imshow('origin', img)
    CV2.waitKey(0)

    CV2.imshow('new', new_img)
    CV2.waitKey(0)


