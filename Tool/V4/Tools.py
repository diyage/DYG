import torch
import numpy as np
from Tool.BaseTools import BaseTools
from typing import Union


class YOLOV4Tools(BaseTools):

    @staticmethod
    def compute_anchor_response_result(
            anchor_pre_wh: dict,
            grid_number_dict: dict,
            abs_gt_pos: Union[tuple, list],
            image_wh: Union[tuple, list],
            iou_th: float = 0.6,
            multi_gt: bool = False
    ):
        # flatten
        keys = list(anchor_pre_wh.keys())
        values = np.array(list(anchor_pre_wh.values()))
        anchor_pre_wh = values.reshape(-1, 2).tolist()

        grid_number_ = np.array(list(grid_number_dict.values())).repeat(3, axis=1)
        grid_number_ = grid_number_.reshape(-1, 2).tolist()
        # ----------------------------------------------------------------------
        response = [False for _ in range(len(anchor_pre_wh))]
        best_index = 0
        best_iou = 0

        weight_vec = []
        iou_vec = []
        gt_w = abs_gt_pos[2] - abs_gt_pos[0]
        gt_h = abs_gt_pos[3] - abs_gt_pos[1]

        if gt_w < 1e-4 or gt_h < 1e-4:
            # valid obj box
            return None

        s1 = gt_w * gt_h
        for index, val in enumerate(anchor_pre_wh):

            grid_number = grid_number_[index]

            anchor_w = val[0] / grid_number[0] * image_wh[0]  # scaled on image
            anchor_h = val[1] / grid_number[1] * image_wh[1]  # scaled on image

            s0 = anchor_w * anchor_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            union = s0 + s1 - inter
            iou = inter / (union + 1e-8)

            if multi_gt:
                if iou > iou_th:
                    response[index] = True
                else:
                    pass
            else:
                if iou >= best_iou:
                    best_index = index
                    best_iou = iou
                else:
                    pass

            weight_vec.append(
                2.0 - (gt_w / image_wh[0]) * (gt_h / image_wh[1])
            )
            iou_vec.append(iou)

        if multi_gt:
            # in multi_ground_truth, (at least) one(or more) anchor(s) will response
            for iou_index in range(len(iou_vec)):
                if not response[iou_index]:
                    weight_vec[iou_index] = 0.0  # negative anchor
        else:
            for iou_index in range(len(iou_vec)):
                if iou_index != best_index:
                    if iou_vec[iou_index] >= iou_th:
                        weight_vec[iou_index] = - 1.0  # ignore this anchor
                    else:
                        weight_vec[iou_index] = 0.0  # negative anchor

        # ----------------------------------------------------------------------
        w_v = np.array(weight_vec).reshape(values.shape[0], values.shape[1]).tolist()
        # (3, 3)
        res = {}
        for anchor_index, anchor_key in enumerate(keys):
            res[anchor_key] = w_v[anchor_index]

        return res

    @staticmethod
    def make_target(
            labels: list,
            anchor_pre_wh: dict,
            image_wh: tuple,
            grid_number: dict,
            kinds_name: list,
            iou_th: float = 0.5,
            multi_gt: bool = False
    ):
        '''

                Args:
                    labels: [label0, label1, ...]
                            label --> [obj0, obj1, ...]
                            obj --> [kind_name, x, y, x, y]  not scaled
                    anchor_pre_wh:  key --> "for_s", "for_m", "for_l"
                    image_wh:
                    grid_number: key --> "for_s", "for_m", "for_l"
                    kinds_name:
                    iou_th:
                    multi_gt:

                Returns:
                    {
                        "for_s": (N, a_n, 5+k_n, s, s) --> (N, -1, s, s)
                        "for_m": (N, a_n, 5+k_n, m, m) --> (N, -1, m, m)
                        "for_l": (N, a_n, 5+k_n, l, l) --> (N, -1, l, l)
                    }

                '''
        kinds_number = len(kinds_name)
        N = len(labels)
        res = {}
        for anchor_key, val in grid_number.items():
            a_n, H, W = len(anchor_pre_wh[anchor_key]), val[1], val[0]
            res[anchor_key] = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(labels):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[0])
                abs_pos = obj[1:]

                weight_dict = YOLOV4Tools.compute_anchor_response_result(
                    anchor_pre_wh,
                    grid_number,
                    abs_pos,
                    image_wh,
                    iou_th,
                    multi_gt
                )
                # weight_dict : key --> anchor_key, value --> weight of pre_anchor with gt_obj
                # (3, 3)
                if weight_dict is None:
                    continue

                pos = [val / image_wh[0] for val in abs_pos]  # scaled in [0, 1]

                for anchor_key in anchor_pre_wh.keys():
                    weight_vec = weight_dict[anchor_key]  # weight_vec of one anchor size
                    grid_size = (
                        image_wh[0] // grid_number[anchor_key][0],
                        image_wh[1] // grid_number[anchor_key][1]
                    )

                    grid_index = (
                        int((abs_pos[0] + abs_pos[2]) * 0.5 // grid_size[0]),  # w -- on x-axis
                        int((abs_pos[1] + abs_pos[3]) * 0.5 // grid_size[1])  # h -- on y-axis
                    )
                    for weight_index, weight_value in enumerate(weight_vec):
                        res[anchor_key][batch_index, weight_index, 4, grid_index[1], grid_index[0]] = weight_value
                        if weight_value != -1 and weight_value != 0:
                            res[anchor_key][batch_index, weight_index, 0:4, grid_index[1], grid_index[0]] = torch.tensor(pos)
                            res[anchor_key][batch_index, weight_index, int(5 + kind_int), grid_index[1], grid_index[0]] = 1.0

        for anchor_key, val in grid_number.items():
            H, W = val[1], val[0]
            res[anchor_key] = res[anchor_key].view(N, -1, H, W)
        return res

    @staticmethod
    def split_target(
            *args,
            **kwargs
    ) -> dict:
        pass

    @staticmethod
    def split_predict(
            *args,
            **kwargs
    ) -> dict:
        pass

    @staticmethod
    def mish(
            x: torch.Tensor
    ) -> torch.Tensor:
        return x*torch.tanh(x*torch.log(1+torch.exp(x)))
    
