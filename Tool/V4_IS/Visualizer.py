import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Union
from Tool.V4_IS.Predictor import YOLOV4PredictorIS
from Tool.V4_IS.Tools import YOLOV4ToolsIS
from Tool.V4_IS.Model import YOLOV4ForISModel
from Tool.BaseTools import CV2, BaseVisualizer
import os


class YOLOV4VisualizerIs(BaseVisualizer):
    def __init__(
            self,
            model: YOLOV4ForISModel,
            predictor: YOLOV4PredictorIS,
            class_colors: list,
            iou_th_for_make_target: float,
            multi_gt: bool
    ):
        super().__init__(
            model,
            predictor,
            class_colors,
            iou_th_for_make_target
        )

        self.predictor = predictor

        self.anchor_keys = self.predictor.anchor_keys
        self.multi_gt = multi_gt

    def change_image_wh(
            self,
            image_wh: tuple
    ):
        self.image_size = image_wh
        self.grid_number, self.pre_anchor_w_h = YOLOV4ToolsIS.get_grid_number_and_pre_anchor_w_h(
            self.image_size,
            self.image_shrink_rate,
            self.pre_anchor_w_h_rate
        )

    def make_targets(
            self,
            labels,
    ):
        targets = YOLOV4ToolsIS.make_target(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            self.iou_th_for_make_target,
            multi_gt=self.multi_gt
        )
        targets['mask'] = targets['mask'].to(self.device)
        for anchor_key in self.anchor_keys:
            targets[anchor_key] = targets[anchor_key].to(self.device)
        return targets

    def detect_one_image(
            self,
            image: Union[torch.Tensor, np.ndarray],
            saved_path: str,
    ):
        print('I have not implement this method')

    def visualize(
            self,
    ):
        pass

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        os.makedirs(saved_dir, exist_ok=True)
        for batch_id, (images, objects, masks) in enumerate(tqdm(data_loader_test,
                                                                 desc=desc,
                                                                 position=0)):
            if batch_id == 10:
                break

            self.detector.eval()
            images = images.to(self.device)

            labels = [objects, masks]
            targets = self.make_targets(labels)

            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets)  # [kps_vec, masks_vec]_s
            pre_decode = self.predictor.decode_predict(output)  # [kps_vec, masks_vec]_s

            for image_index in range(images.shape[0]):
                self.visualize()


