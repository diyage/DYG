from Tool.V4_IS.Predictor import YOLOV4PredictorIS
from Tool.V4.FormalEvaluator import YOLOV4FormalEvaluator
from Tool.V4_IS.Model import YOLOV4ForISModel
import torch
import numpy as np


class YOLOV4FormalEvaluatorIS(YOLOV4FormalEvaluator):
    def __init__(
            self,
            model: YOLOV4ForISModel,
            predictor: YOLOV4PredictorIS,
            data_root: str,
            img_size: int,
            device: str,
            transform,
            labelmap: list,
            display=False,
            use_07: bool = True
    ):
        super().__init__(
            model,
            predictor,
            data_root,
            img_size,
            device,
            transform,
            labelmap,
            display,
            use_07
        )

    def get_predict_info(
            self,
            net,
            x: torch.Tensor,

    ) -> dict:
        assert len(x.shape) == 4 and x.shape[0] == 1

        out = net(x)
        kps_vec = self.predictor.decode_one_predict(out)[0]
        """
                decode_one_predict(out)[0]   --> kps_vec
                decode_one_predict(out)[1]   --> mask_vec  (ignore)
                Please see method --> YOLOV4PredictorIS.decode_one_predict
        """

        bboxes = []
        scores = []
        cls_inds = []

        for kps in kps_vec:
            bboxes.append(kps[1])
            scores.append(kps[2])
            cls_inds.append(self.labelmap.index(kps[0]))
        if len(bboxes) != 0:
            bboxes = np.array(bboxes) / self.img_size
        else:
            bboxes = np.array(bboxes)
        scores = np.array(scores)
        cls_inds = np.array(cls_inds)

        return {
            'bboxes': bboxes,
            'scores': scores,
            'cls_inds': cls_inds
        }
