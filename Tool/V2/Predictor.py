import torch
from .Tools import YOLOV2Tools
from Tool.BaseTools import BasePredictor
import torch.nn.functional as F


class YOLOV2Predictor(BasePredictor):
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
        super().__init__(
            iou_th,
            prob_th,
            conf_th,
            score_th,
            pre_anchor_w_h,
            kinds_name,
            image_size,
            grid_number
        )

    def decode_out_one_image(
            self,
            out_put: torch.Tensor,
            out_is_target: bool = False,
    ) -> list:
        '''

        Args:
            out_put: (_, H, W) or (H, W, a_n, 11)
            out_is_target: bool

        Returns: kps_s  ---> kps = (kind_name, (x, y, x, y), score)
            [
            (kind_name, (x, y, x, y), score),
            (kind_name, (x, y, x, y), score)
             ...
             ]

        '''
        if out_is_target:
            assert len(out_put.shape) == 4
        else:
            assert len(out_put.shape) == 3

        out_put = out_put.unsqueeze(dim=0)
        a_n = len(self.pre_anchor_w_h)

        res_dict = YOLOV2Tools.split_output(
            out_put,
            a_n,
            is_target=out_is_target
        )

        if not out_is_target:
            conf = torch.sigmoid(res_dict.get('conf'))
            cls_prob = torch.softmax(res_dict.get('cls_prob'), dim=-1)
            position = res_dict.get('position')[0]
            position_abs = YOLOV2Tools.xywh_to_xyxy(
                position,
                self.pre_anchor_w_h,
                self.image_size,
                self.grid_number
            ).clamp_(0, self.image_size[0]-1)
            # scaled on image
        else:
            conf = res_dict.get('conf')
            cls_prob = F.one_hot(res_dict.get('cls_ind').long(), len(self.kinds_name))

            position_abs = res_dict.get('position')[1] * self.image_size[0]
            # scaled on image

        position_abs_ = position_abs.contiguous().view(-1, 4)
        conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
        cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
        scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
        # (-1, kinds_num)

        cls_prob_mask = cls_prob_.max(dim=-1)[0] > self.prob_th   # type: torch.Tensor
        # (-1, )

        conf_mask = conf_ > self.conf_th  # type: torch.Tensor
        # (-1, )

        scores_max_value, scores_max_index = scores_.max(dim=-1)
        # (-1, )

        scores_mask = scores_max_value > self.score_th  # type: torch.Tensor
        # (-1, )

        mask = (conf_mask.float() * cls_prob_mask.float() * scores_mask.float()).bool()
        # (-1, )

        return self.nms(
            position_abs_[mask],
            scores_max_value[mask],
            scores_max_index[mask]
        )
