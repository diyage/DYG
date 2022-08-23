from Tool.V4 import YOLOV4Tools
import torch
from typing import *
import numpy as np


class YOLOV4ToolsIS(YOLOV4Tools):
    def __init__(self):
        super().__init__()

    @staticmethod
    def make_target(
            objects_vec: List[List[List]],
            masks_vec: List[List[np.ndarray]],
            anchor_pre_wh: dict,
            image_wh: tuple,
            grid_number: dict,
            kinds_name: list,
            iou_th: float = 0.5,
            multi_gt: bool = False
    ):

        kinds_number = len(kinds_name)
        N = len(objects_vec)
        res = {}
        for anchor_key, val in grid_number.items():
            a_n, H, W = len(anchor_pre_wh[anchor_key]), val[1], val[0]
            res[anchor_key] = torch.zeros(size=(N, a_n, 5 + kinds_number, H, W))

        for batch_index, label in enumerate(objects_vec):  # an image label
            for obj_index, obj in enumerate(label):  # many objects
                kind_int = kinds_name.index(obj[-1])
                abs_pos = obj[:4]

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
                    # this grid_index in top-left index
                    # in yolo v4, we use top-left, top-right, down-left, down-right, four grid_index(s)
                    grid_index_vec = [grid_index]

                    if grid_index[0] + 1 < grid_number[anchor_key][0]:
                        grid_index_vec.append(
                            (grid_index[0] + 1, grid_index[1])
                        )

                    if grid_index[1] + 1 < grid_number[anchor_key][1]:
                        grid_index_vec.append(
                            (grid_index[0], grid_index[1] + 1)
                        )

                    if grid_index[0] + 1 < grid_number[anchor_key][0] and grid_index[1] + 1 < grid_number[anchor_key][
                        1]:
                        grid_index_vec.append(
                            (grid_index[0] + 1, grid_index[1] + 1)
                        )

                    for weight_index, weight_value in enumerate(weight_vec):
                        for grid_index_x, grid_index_y in grid_index_vec:
                            res[anchor_key][batch_index, weight_index, 4, grid_index_y, grid_index_x] = weight_value
                            if weight_value != -1 and weight_value != 0:
                                res[anchor_key][batch_index, weight_index, 0:4, grid_index_y, grid_index_x] = torch.tensor(pos)
                                res[anchor_key][batch_index, weight_index, int(5 + kind_int), grid_index_y, grid_index_x] = 1.0

        for anchor_key, val in grid_number.items():
            H, W = val[1], val[0]
            res[anchor_key] = res[anchor_key].view(N, -1, H, W)

        res['mask'] = torch.tensor(masks_vec, dtype=torch.float32)
        return res

    @staticmethod
    def split_target(
            target: dict,
            anchor_number_for_single_size: int,
    ) -> dict:
        res = {}
        # key : 'for_s', 'for_m', 'for_l', 'mask'
        # val : dict --> {'position': xxx, 'conf': xxx, 'cls_prob': xxx}
        for key, x in target.items():
            if key == 'mask':
                # from  (-1, kinds_num + 1, h, w) to (-1, h, w, kinds_num + 1)
                x = x.permute(0, 2, 3, 1)
                res[key] = x
            else:
                N, C, H, W = x.shape
                K = C // anchor_number_for_single_size
                # K = (x, y, w, h, conf, kinds0, kinds1, ...)
                # C = anchor_number * K
                x = x.view(N, anchor_number_for_single_size, K, H, W)
                x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

                position = [None, x[..., 0:4]]  # scaled in [0, 1]
                conf = x[..., 4]  # N * H * W * a_n
                cls_prob = x[..., 5:]  # N * H * W * a_n * ...

                now_size_res = {
                    'position': position,  # first txty_twth, second xyxy(scaled in [0, 1])
                    'conf': conf,
                    'cls_prob': cls_prob
                }
                res[key] = now_size_res

        return res

    @staticmethod
    def split_predict(
            out_put: dict,
            anchor_number_for_single_size: int,
    ) -> dict:
        res = {}
        # key : 'for_s', 'for_m', 'for_l', 'mask'
        # val : dict --> {'position': xxx, 'conf': xxx, 'cls_prob': xxx}
        for key, x in out_put.items():
            if key == 'mask':
                # from  (-1, kinds_num + 1, h, w) to (-1, h, w, kinds_num + 1)
                x = x.permute(0, 2, 3, 1)
                res[key] = x
            else:
                N, C, H, W = x.shape
                K = C // anchor_number_for_single_size  # K = (x, y, w, h, conf, kinds0, kinds1, ...)
                # C = anchor_number * K
                x = x.view(N, anchor_number_for_single_size, K, H, W)
                x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

                position = [x[..., 0:4], None]
                conf = x[..., 4]  # N * H * W * a_n
                cls_prob = x[..., 5:]  # N * H * W * a_n * ...

                now_size_res = {
                    'position': position,  # first txty_twth, second xyxy(scaled in [0, 1])
                    'conf': conf,
                    'cls_prob': cls_prob
                }
                res[key] = now_size_res

        return res
