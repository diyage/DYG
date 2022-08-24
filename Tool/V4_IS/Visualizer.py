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
from typing import List


class YOLOV4VisualizerIs(BaseVisualizer):
    def __init__(
            self,
            model: YOLOV4ForISModel,
            predictor: YOLOV4PredictorIS,
            class_colors: list,
            iou_th_for_make_target: float,
            multi_gt: bool,
            image_mean: List[float],
            image_std: List[float],
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
        self.image_mean = image_mean
        self.image_std = image_std

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

    def show(
            self,
            image: np.ndarray,
            decode_detection: List,
            mask_vec: np.ndarray,
            saved_file_name: str
    ):
        image = image.copy().astype(np.float32)
        mask_vec = mask_vec.copy().astype(np.float32)

        h, w, _ = image.shape

        for d in decode_detection:
            predict_kind_name, abs_double_pos, prob_score = d
            """
                draw bbox(es) on image
            """
            color = self.class_colors[self.kinds_name.index(predict_kind_name)]
            x0, y0 = int(abs_double_pos[0]), int(abs_double_pos[1])
            x0 = max(int(0), x0)
            y0 = max(int(0), y0)

            x1, y1 = int(abs_double_pos[2]), int(abs_double_pos[3])
            x1 = min(int(w - 1), x1)
            y1 = min(int(h - 1), y1)

            CV2.rectangle(image,
                          start_point=(x0, y0),
                          end_point=(x1, y1),
                          color=color,
                          thickness=2)

            scale = 0.5
            CV2.putText(image,
                        '{}:{:.2%}'.format(predict_kind_name, prob_score),
                        org=(x0, int(y0 - 5)),
                        font_scale=scale,
                        color=(0, 0, 0),
                        back_ground_color=color
                        )

            """
                cut mask(s) with bbox(es)
            """
            kind_index = self.kinds_name.index(predict_kind_name)
            mask_index = kind_index + 1  # mask index 0 is used for background
            now_kind_mask = mask_vec[:, :, mask_index]  # (h, w)
            mask_keep_region = np.zeros_like(now_kind_mask)
            mask_keep_region[y0:y1, x0:x1] = 1.0
            mask_ = now_kind_mask * mask_keep_region
            """
                        draw cuted mask(s) on image.
                        actually, mask is semantic segmentation mask. 
                        after, cuted, instance segmentation mask !
            """
            color = [np.random.randint(255) for _ in range(3)]

            for c in range(3):
                image[..., c] = np.where(
                    mask_ == 1.0,
                    0.5 * mask_ * color[c] + 0.5 * image[..., c],
                    image[..., c]
                )

        CV2.imwrite(saved_file_name, image)

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        os.makedirs(saved_dir, exist_ok=True)
        for batch_ind, (images, objects, masks) in enumerate(tqdm(data_loader_test,
                                                                  desc=desc,
                                                                  position=0)):
            if batch_ind == 10:
                break

            self.detector.eval()
            images = images.to(self.device)

            labels = [objects, masks]
            targets = self.make_targets(labels)

            output = self.detector(images)

            gt_decode = self.predictor.decode_target(targets)  # [kps_vec, masks_vec]_s
            pre_decode = self.predictor.decode_predict(output)  # [kps_vec, masks_vec]_s

            for image_ind in range(images.shape[0]):
                image_i = images[image_ind].permute(1, 2, 0).cpu().detach().numpy()
                image_i = image_i * np.array(self.image_std) + np.array(self.image_mean)
                image_i = image_i * 255.0
                image_i = image_i.astype(np.uint8)

                pre_decode_detection, pre_decode_mask = pre_decode[image_ind][0], pre_decode[image_ind][1]

                gt_decode_detection, gt_decode_mask = gt_decode[image_ind][0], gt_decode[image_ind][1]

                self.show(
                    image_i,
                    pre_decode_detection,
                    pre_decode_mask,
                    saved_file_name='{}/{}_{}_pre.png'.format(saved_dir, batch_ind, image_ind)
                )
                self.show(
                    image_i,
                    gt_decode_detection,
                    gt_decode_mask,
                    saved_file_name='{}/{}_{}_gt.png'.format(saved_dir, batch_ind, image_ind)
                )





