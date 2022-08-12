'''
This packet is used for visualization.
'''
from abc import abstractmethod
from .predictor import BasePredictor
from .model import BaseModel


class BaseVisualizer:
    def __init__(
            self,
            model: BaseModel,
            predictor: BasePredictor,
            class_colors: list,
            iou_th_for_make_target: float
    ):
        self.detector = model  # type: BaseModel
        self.device = next(model.parameters()).device

        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th_for_show = self.predictor.iou_th

        self.class_colors = class_colors
        self.iou_th_for_make_target = iou_th_for_make_target

    @abstractmethod
    def make_targets(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def detect_one_image(
            self,
            *args,
            **kwargs
    ):
        pass


