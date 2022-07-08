import torch
from V1.tools.position_translate import PositionTranslate


class YOLOV1Tools:
    def __init__(self,
                 kinds_name: list,
                 grid_number: tuple = (7, 7),
                 image_wh: tuple = (448, 448),
                 ):
        self.grid_number = grid_number
        self.image_wh = image_wh
        self.kinds_name = kinds_name
        self.kinds_number = len(kinds_name)

    def make_targets(self, labels: list):
        shape = (len(labels), self.grid_number[1], self.grid_number[0],  10 + self.kinds_number)
        target = torch.zeros(size=shape)
        # (x, y, w, h, c4) (x, y, w, h, c9) (class1, class2, ...)
        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = self.kinds_name.index(obj[0])
                pos_trans = PositionTranslate(obj[1:], types='abs_double',
                                              image_size=self.image_wh,
                                              grid_number=self.grid_number)
                offset_position = pos_trans.center_offset_position.get_position()  # type: tuple

                grid_index = pos_trans.grid_index_to_x_y_axis  # type: tuple

                target[batch_index, grid_index[1], grid_index[0], int(10 + kind_int)] = 1
                target[batch_index, grid_index[1], grid_index[0], 0:4] = torch.tensor(offset_position)
                target[batch_index, grid_index[1], grid_index[0], 5:9] = torch.tensor(offset_position)
                target[batch_index, grid_index[1], grid_index[0], [4, 9]] = 1

        return target

    def compute_box_iou(self, box1, box2, grid_index: tuple) -> float:
        p1 = PositionTranslate(p=(box1[0], box1[1], box1[2], box1[3]),
                               types='center_offset',
                               image_size=self.image_wh,
                               grid_index=grid_index)
        p2 = PositionTranslate(p=(box2[0], box2[1], box2[2], box2[3]),
                               types='center_offset',
                               image_size=self.image_wh,
                               grid_index=grid_index)

        im1 = p1.abs_double_position
        im2 = p2.abs_double_position

        s1 = (im1.m - im1.a) * (im1.n - im1.b)
        s2 = (im2.m - im2.a) * (im2.n - im2.b)
        a = max(im1.a, im2.a)
        b = max(im1.b, im2.b)

        m = min(im1.m, im2.m)
        n = min(im1.n, im2.n)

        w = max(m - a, 0)
        h = max(n - b, 0)
        inter = w * h
        union = s1 + s2 - inter
        return 1.0 * inter / union

    def compute_boxs_iou(
            self,
            boxs1: list,
            boxs2: list,
            grid_index: tuple
    ) -> list:
        m = len(boxs1)
        n = len(boxs2)
        res = [[0] * n for _ in range(m)]

        for i, box1 in enumerate(boxs1):
            for j, box2 in enumerate(boxs2):
                res[i][j] = self.compute_box_iou(box1, box2, grid_index)

        return res
