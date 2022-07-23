import torch
from Tool.BaseTools.tools import BaseTools


class BasePredictor:
    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            score_th: float,
            pre_anchor_w_h: tuple,
            kinds_name: list,
            image_size: tuple,
            grid_number: tuple
    ):
        self.iou_th = iou_th
        self.prob_th = prob_th
        self.conf_th = conf_th
        self.score_th = score_th
        self.kinds_name = kinds_name
        self.image_size = image_size
        self.grid_number = grid_number
        self.pre_anchor_w_h = pre_anchor_w_h

    def nms(
            self,
            position_abs_: torch.Tensor,
            scores_max_value: torch.Tensor,
            scores_max_index: torch.Tensor
    ):
        '''
        for one image, all predicted objects( already mask some bad ones)
        it may have many kinds...
        Args:
            position_abs_: (P, 4)
            scores_max_value: (P, ) predicted kind_name's score
            scores_max_index: (P, ) predicted kind_name's index

        Returns: kps_s
            kps_s --> [kps0, kps1, kps2, ...]
            kps --> (predict_kind_name, abs_double_pos, score)
            abs_double_pos --> (x, y, x, y)

        '''
        def for_response(
                now_kind_pos_abs,
                now_kind_scores_max_value,
        ):
            res = []
            keep_index = BaseTools.nms(
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
                    # kps
                )

            return res

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

    def make_targets(
            self,
    ):
        pass

    def decode_out_one_image(
            self,
            out_put: torch.Tensor,
            out_is_target: bool = False,
    ) -> list:
        '''
        please overwrite this method!!!
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
        pass

    def decode(
            self,
            out_put: torch.Tensor,
            out_is_target: bool = False,
    ):
        '''

        Args:
            out_put: (N, _, H, W)/(N,H,W,a_n,11)
            out_is_target: bool

        Returns: [kps_s, kps_s, ...]
            [
               [(kind_name, (x, y, x, y), score), ... ],    --> image one
              [ ... ],           --> image two
            ...
        ]

        '''
        res = []
        for i in range(out_put.shape[0]):
            pre_ = self.decode_out_one_image(
                out_put[i],
                out_is_target
            )
            res.append(pre_)

        return res
