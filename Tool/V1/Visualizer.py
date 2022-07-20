import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Union
from .Predictor import YOLOV1Predictor
from .Tools import YOLOV1Tools


class YOLOV1Visualizer:
    def __init__(
            self,
            model: nn.Module,
            predictor: YOLOV1Predictor,
            class_colors: list
    ):
        self.detector = model  # type: nn.Module
        self.detector.cuda()

        self.predictor = predictor
        self.pre_anchor_w_h = self.predictor.pre_anchor_w_h
        self.image_size = self.predictor.image_size
        self.grid_number = self.predictor.grid_number
        self.kinds_name = self.predictor.kinds_name
        self.iou_th = self.predictor.iou_th
        self.class_colors = class_colors

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ) -> torch.Tensor:
        return YOLOV1Tools.make_targets(
            labels,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            need_abs
        )

    def detect_one_image(
            self,
            image: Union[torch.Tensor, np.ndarray],
            saved_path: str
    ):
        if isinstance(image, np.ndarray):
            image = YOLOV1Tools.image_np_to_tensor(image)

        out = self.detector(image.unsqueeze(0).cuda())[0]
        pre_kps_s = self.predictor.decode_out_one_image(
            out,
            out_is_target=False
        )
        YOLOV1Tools.visualize(
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
            images = images.cuda()
            targets = self.make_targets(labels, need_abs=True).cuda()
            output = self.detector(images)

            gt_decode = self.predictor.decode(targets, out_is_target=True)
            pre_decode = self.predictor.decode(output, out_is_target=False)

            for image_index in range(images.shape[0]):

                YOLOV1Tools.visualize(
                    images[image_index],
                    gt_decode[image_index],
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )

                YOLOV1Tools.visualize(
                    images[image_index],
                    pre_decode[image_index],
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                    class_colors=self.class_colors,
                    kinds_name=self.kinds_name
                )
            break
