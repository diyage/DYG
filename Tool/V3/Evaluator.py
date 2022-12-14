from .Predictor import YOLOV3Predictor
from .Tools import YOLOV3Tools
from .Model import YOLOV3Model
from Tool.BaseTools import BaseEvaluator


class YOLOV3Evaluator(BaseEvaluator):
    def __init__(
            self,
            model: YOLOV3Model,
            predictor: YOLOV3Predictor,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            predictor,
            iou_th_for_make_target
        )

        self.predictor = predictor
        self.anchor_keys = self.predictor.anchor_keys

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV3Tools.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels
    ):
        targets = YOLOV3Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets
