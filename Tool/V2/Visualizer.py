import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Union
from .Predictor import YOLOV2Predictor
from .Tools import YOLOV2Tools
from .Model import YOLOV2Model
from Tool.BaseTools import CV2, BaseVisualizer
import os


class YOLOV2Visualizer(BaseVisualizer):
    def __init__(
            self,
            model: YOLOV2Model,
            predictor: YOLOV2Predictor,
            class_colors: list,
            iou_th_for_make_target: float
    ):
        super().__init__(
            model,
            predictor,
            class_colors,
            iou_th_for_make_target
        )

    def make_targets(
            self,
            labels,
    ):
        return YOLOV2Tools.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
            ).to(self.device)

    def detect_one_image(
            self,
            image: Union[torch.Tensor, np.ndarray],
            saved_path: str,
    ):
        if isinstance(image, np.ndarray):
            image = CV2.resize(image, new_size=(416, 416))
            print('We resize the image to (416, 416), that may not be what you want!' +
                  'please resize your image before using this method!')
            image = YOLOV2Tools.image_np_to_tensor(image)

        out = self.detector(image.unsqueeze(0).to(self.device))  # (1, 3, H, W)
        pre_kps_s = self.predictor.decode_one_predict(
            out,
        )
        YOLOV2Tools.visualize(
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
        os.makedirs(saved_dir, exist_ok=True)
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            if batch_id == 10:
                break

            self.detector.eval()
            images = images.to(self.device)
            targets = self.make_targets(labels)
            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets)
            pre_decode = self.predictor.decode_predict(output)

            for image_index in range(images.shape[0]):

                YOLOV2Tools.visualize(
                    images[image_index],
                    gt_decode[image_index],
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                YOLOV2Tools.visualize(
                    images[image_index],
                    pre_decode[image_index],
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )
