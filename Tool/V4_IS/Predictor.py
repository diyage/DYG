import torch
from Tool.V4_IS.Tools import YOLOV4ToolsIS
from Tool.V4 import YOLOV4Predictor
from typing import List
import numpy as np


class YOLOV4PredictorIS(YOLOV4Predictor):
    def __init__(
            self,
            iou_th: float,
            prob_th: float,
            conf_th: float,
            score_th: float,
            pre_anchor_w_h_rate: dict,
            kinds_name: list,
            image_size: tuple,
            image_shrink_rate: dict,
            each_size_anchor_number: int = 3,
            segmentation_mask_threshold: float = 0.5
    ):
        super().__init__(
            iou_th,
            prob_th,
            conf_th,
            score_th,
            pre_anchor_w_h_rate,
            kinds_name,
            image_size,
            image_shrink_rate,
            each_size_anchor_number,
        )
        self.segmentation_mask_threshold = segmentation_mask_threshold

    def decode_one_target(
            self,
            x: dict,
    ) -> List:
        '''

        Args:
            x: key --> ["for_s", "for_m", "for_l", 'mask']
               val --> (1, -1, H, W)

        Returns:

        '''

        a_n = self.each_size_anchor_number
        res_target = YOLOV4ToolsIS.split_target(
            x,
            a_n
        )

        masked_pos_vec = []
        masked_score_max_value_vec = []
        masked_score_max_index_vec = []

        for anchor_key in self.anchor_keys:
            res_dict = res_target[anchor_key]
            # -------------------------------------------------------------------
            conf = (res_dict.get('conf') > 0).float()
            cls_prob = res_dict.get('cls_prob')
            position_abs = res_dict.get('position')[1]  # scaled in [0, 1]
            position_abs = position_abs * self.image_size[0]   # scaled on image
            # -------------------------------------------------------------------
            position_abs_ = position_abs.contiguous().view(-1, 4)
            conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
            cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
            scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
            # (-1, kinds_num)
            # -------------------------------------------------------------------
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
            # -------------------------------------------------------------------
            masked_pos_vec.append(position_abs_[mask])
            masked_score_max_value_vec.append(scores_max_value[mask]),
            masked_score_max_index_vec.append(scores_max_index[mask])

        kps_vec = self.nms(
            torch.cat(masked_pos_vec, dim=0),
            torch.cat(masked_score_max_value_vec, dim=0),
            torch.cat(masked_score_max_index_vec, dim=0)
        )

        gt_mask_vec = res_target['mask'].squeeze(0).cpu().detach().numpy()

        return [kps_vec, gt_mask_vec]

    def decode_target(
            self,
            target: dict,
    ) -> List[List]:
        batch_size = target[self.anchor_keys[0]].shape[0]
        res = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            tmp = {}
            for anchor_key in self.anchor_keys:
                tmp[anchor_key] = target[anchor_key][i].unsqueeze(0)

            tmp['mask'] = target['mask'][i].unsqueeze(0)

            pre_ = self.decode_one_target(tmp)
            res[i] += pre_

        return res

    def decode_one_predict(
            self,
            x: dict
    ) -> List:
        '''

                Args:
                    x: key --> ["for_s", "for_m", "for_l", 'mask']
                       val --> (1, -1, H, W)

                Returns:

                '''

        a_n = self.each_size_anchor_number
        res_out = YOLOV4ToolsIS.split_predict(
            x,
            a_n
        )
        masked_pos_vec = []
        masked_score_max_value_vec = []
        masked_score_max_index_vec = []

        for anchor_key in self.anchor_keys:
            res_dict = res_out[anchor_key]
            # -------------------------------------------------------------------
            conf = torch.sigmoid(res_dict.get('conf'))
            cls_prob = torch.softmax(res_dict.get('cls_prob'), dim=-1)
            position = res_dict.get('position')[0]  # txtytwth
            position_abs = YOLOV4ToolsIS.txtytwth_to_xyxy(
                position,
                self.pre_anchor_w_h.get(anchor_key),
                self.grid_number.get(anchor_key),
            ).clamp_(0, 1)  # scaled in [0, 1]
            position_abs = position_abs * self.image_size[0]  # scaled on image

            # -------------------------------------------------------------------
            position_abs_ = position_abs.contiguous().view(-1, 4)
            conf_ = conf.contiguous().view(-1, )  # type: torch.Tensor
            cls_prob_ = cls_prob.contiguous().view(-1, len(self.kinds_name))
            scores_ = cls_prob_ * conf_.unsqueeze(-1).expand_as(cls_prob_)
            # (-1, kinds_num)

            # -------------------------------------------------------------------

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

            # -------------------------------------------------------------------
            masked_pos_vec.append(position_abs_[mask])
            masked_score_max_value_vec.append(scores_max_value[mask]),
            masked_score_max_index_vec.append(scores_max_index[mask])

        kps_vec = self.nms(
            torch.cat(masked_pos_vec, dim=0),
            torch.cat(masked_score_max_value_vec, dim=0),
            torch.cat(masked_score_max_index_vec, dim=0)
        )
        pre_mask_vec = torch.sigmoid(res_out['mask']).squeeze(0).cpu().detach().numpy().copy()
        pre_mask_vec = np.where(pre_mask_vec > self.segmentation_mask_threshold, 1.0, 0.0)
        return [kps_vec, pre_mask_vec]

    def decode_predict(
            self,
            predict: dict,
    ) -> List[List]:
        batch_size = predict[self.anchor_keys[0]].shape[0]
        res = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            tmp = {}
            for anchor_key in self.anchor_keys:
                tmp[anchor_key] = predict[anchor_key][i].unsqueeze(0)

            tmp['mask'] = predict['mask'][i].unsqueeze(0)
            pre_ = self.decode_one_predict(tmp)

            res[i] += pre_

        return res


def debug_decode_out():
    from Tool.V4_IS.Model import YOLOV4ForISModel, CSPDarkNet53IS
    from Tool.V4 import YOLOV4Config
    config = YOLOV4Config()

    x = torch.rand(size=(1, 3, 608, 608))
    backbone = CSPDarkNet53IS()
    net = YOLOV4ForISModel(backbone, 3, 20)
    out = net(x)

    predictor = YOLOV4PredictorIS(
        0.5,
        0.0,
        0.0,
        0.1,
        config.data_config.pre_anchor_w_h_rate,
        config.data_config.kinds_name,
        config.data_config.image_size,
        config.data_config.image_shrink_rate,
    )

    decode_out = predictor.decode_one_predict(out)
    print(len(decode_out))
    for kps in decode_out[0]:
        print(kps)
    print(decode_out[1].shape)


if __name__ == '__main__':
    debug_decode_out()