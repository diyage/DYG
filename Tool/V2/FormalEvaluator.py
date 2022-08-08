from .Predictor import YOLOV2Predictor
from Tool.BaseTools import BaseFormalEvaluator


class YOLOV2FormalEvaluator(BaseFormalEvaluator):
    def __init__(
            self,
            model,
            predictor: YOLOV2Predictor,
            data_root: str,
            img_size: int,
            device: str,
            transform,
            labelmap: list,
            display=False,
            use_07: bool = True
    ):
        super().__init__(
            predictor,
            data_root,
            img_size,
            device,
            transform,
            labelmap,
            display,
            use_07
        )
        self.model = model

    def eval_detector_mAP(
            self,
    ):
        self.evaluate(self.model)
