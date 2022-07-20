import torch
from .Tools import YOLOV1Tools
from Tool.BaseTools import BasePredictor


class YOLOV1Predictor(BasePredictor):
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
            out_put: (_, H, W)
            out_is_target: bool

        Returns: kps_s(what is kps_s, please self.nms() --> base method )
        '''

        assert len(out_put.shape) == 3
        #  _ * H * W
        out_put = out_put.unsqueeze(dim=0)
        #  1 * _ * H * W

        position, conf, cls_prob = YOLOV1Tools.split_output(
            out_put,
        )

        if not out_is_target:
            position = torch.sigmoid(position)
            conf = torch.sigmoid(conf)
            cls_prob = torch.softmax(cls_prob, dim=-1)

            position_abs = YOLOV1Tools.xywh_to_xyxy(
                position,
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

        cls_prob_mask = cls_prob_.max(dim=-1)[0] > self.prob_th  # type: torch.Tensor
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


