import torch
import torch.nn as nn
from .BaseTools import YOLOV2Tools


class Predictor:
    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            pre_anchor_w_h: tuple,
            kinds_name: list,
            image_size: tuple,
            grid_number: tuple
    ):
        self.pre_anchor_w_h = pre_anchor_w_h
        self.iou_th = iou_th
        self.prob_th = prob_th
        self.conf_th = conf_th
        self.kinds_name = kinds_name
        self.image_size = image_size
        self.grid_number = grid_number

    def __nms(
            self,
            position_abs_: torch.Tensor,
            scores_: torch.Tensor
    ):

        def for_response(
                now_kind_pos_abs,
                now_kind_scores_max_value,
        ):
            res = []
            keep_index = YOLOV2Tools.nms(
                now_kind_pos_abs,
                now_kind_scores_max_value,
                threshold=iou_th,
            )

            for index in keep_index:
                s = now_kind_scores_max_value[index]
                abs_double_pos = tuple(now_kind_pos_abs[index].cpu().detach().numpy().tolist())
                predict_kind_name = kind_name

                res.append(
                    (predict_kind_name, abs_double_pos, s.item())
                )

            return res

        scores_max_value, scores_max_index = scores_.max(dim=-1)

        iou_th = self.iou_th
        kinds_name = self.kinds_name

        total = []
        for kind_index, kind_name in enumerate(kinds_name):
            now_kind_response = scores_max_index == kind_index
            total = total + for_response(
                position_abs_[now_kind_response],
                scores_max_value[now_kind_response],
            )

        return total

    def decode_out_one_image(
            self,
            out_put: torch.Tensor,
            out_is_target: bool = False,
    ) -> list:
        '''

        Args:
            out_put: (_, H, W)
            out_is_target: bool

        Returns: kps_s  ---> kps = (kind_name, (x, y, x, y), score)
            [
            (kind_name, (x, y, x, y), score),
            (kind_name, (x, y, x, y), score)
             ...
             ]

        '''
        assert len(out_put.shape) == 3
        #  _ * H * W
        out_put = out_put.unsqueeze(dim=0)
        #  1 * _ * H * W
        a_n = len(self.pre_anchor_w_h)
        position, conf, cls_prob = YOLOV2Tools.split_output(
            out_put,
            a_n
        )

        if not out_is_target:
            conf = torch.sigmoid(conf)
            cls_prob = torch.softmax(cls_prob, dim=-1)
            position_abs = YOLOV2Tools.xywh_to_xyxy(
                position,
                self.pre_anchor_w_h,
                self.image_size,
                self.grid_number
            )
        else:
            position_abs = position

        position_abs_ = position_abs.contiguous().view(-1, 4)
        conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
        cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
        scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
        # (-1, kinds_num)

        cls_prob_max_value = cls_prob_.max(dim=-1)[0]  # type: torch.Tensor
        # (-1, )
        cls_prob_mask = cls_prob_max_value > self.prob_th   # type: torch.Tensor
        # (-1, )
        conf_mask = conf_ > self.conf_th  # type: torch.Tensor
        # (-1, )

        mask = (conf_mask.float() * cls_prob_mask.float()).bool()

        return self.__nms(
            position_abs_[mask],
            scores_[mask],
        )

    def decode(
            self,
            out_put: torch.Tensor,
            out_is_target: bool = False,
    ):
        '''

        Args:
            out_put: (N, _, H, W)
            out_is_target: bool

        Returns: [kps_s, kps_s, ...]
            [
               [
              (kind_name, (x, y, x, y), score),
              (kind_name, (x, y, x, y), score),
               ...
                    ],    --> image one
            [ ... ],           --> image two
            ...
        ]

        '''
        assert len(out_put.shape) == 4
        res = []
        for i in range(out_put.shape[0]):
            pre_ = self.decode_out_one_image(
                out_put[i],
                out_is_target
            )
            res.append(pre_)

        return res
