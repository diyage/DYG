import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Union
from .Predictor import YOLOV3Predictor
from .Tools import YOLOV3Tools
from Tool.BaseTools import CV2, BaseVisualizer


class YOLOV3Visualizer(BaseVisualizer):
    def __init__(
            self,
            model: nn.Module,
            predictor: YOLOV3Predictor,
            class_colors: list
    ):
        super().__init__()
        self.detector = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.backbone = model.backbone  # type: nn.Module
        # be careful, darknet19 is not the detector
        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th = self.predictor.iou_th
        self.anchor_keys = self.predictor.anchor_keys
        self.class_colors = class_colors

    def make_targets(
            self,
            labels,
    ):
        targets = YOLOV3Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th,
        )
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets

    def detect_one_image(
            self,
            image: Union[torch.Tensor, np.ndarray],
            saved_path: str,
    ):
        if isinstance(image, np.ndarray):
            image = CV2.resize(image, new_size=(416, 416))
            print('We resize the image to (416, 416), that may not be what you want!' +
                  'please resize your image before using this method!')
            image = YOLOV3Tools.image_np_to_tensor(image)

        out_dict = self.detector(image.unsqueeze(0).to(self.device))
        pre_kps_s = self.predictor.decode_one_predict(
            out_dict
        )
        YOLOV3Tools.visualize(
            image,
            pre_kps_s,
            saved_path=''.format(saved_path),
            class_colors=self.class_colors,
            kinds_name=self.kinds_name
        )

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):

        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            if batch_id == 10:
                break

            self.detector.eval()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets, batch_size=images.shape[0])

            pre_decode = self.predictor.decode_predict(output, batch_size=images.shape[0])

            for image_index in range(images.shape[0]):

                YOLOV3Tools.visualize(
                    images[image_index],
                    gt_decode[image_index],
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                YOLOV3Tools.visualize(
                    images[image_index],
                    pre_decode[image_index],
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )
