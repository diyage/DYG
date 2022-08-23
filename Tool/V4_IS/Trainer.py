from Tool.V4_IS.Tools import YOLOV4ToolsIS
from Tool.V4_IS.Model import YOLOV4ForISModel
from Tool.V4_IS.Loss import YOLOV4LossIS
from Tool.BaseTools import BaseTrainer, WarmUpOptimizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from typing import Union


class YOLOV4Trainer(BaseTrainer):
    def __init__(
            self,
            model: YOLOV4ForISModel,
            pre_anchor_w_h_rate: dict,
            image_size: tuple,
            image_shrink_rate: dict,
            kinds_name: list,
            iou_th_for_make_target: float,
            multi_gt: bool,
    ):
        super().__init__(
            model,
            pre_anchor_w_h_rate,
            image_size,
            image_shrink_rate,
            kinds_name,
            iou_th_for_make_target,
        )

        self.anchor_keys = list(pre_anchor_w_h_rate.keys())
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

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_loss_func: YOLOV4LossIS,
            optimizer: Union[torch.optim.Optimizer, WarmUpOptimizer],
            now_epoch: int,
            desc: str = '',
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)

        for batch_id, (images, objects, masks) in enumerate(tqdm(data_loader_train,
                                                                 desc=desc,
                                                                 position=0)):
            optimizer.warm(
                now_epoch,
                batch_id,
                max_batch_ind
            )

            self.detector.train()
            images = images.to(self.device)

            labels = [objects, masks]
            targets = self.make_targets(labels)

            output = self.detector(images)
            loss_res = yolo_loss_func(output, targets)
            if not isinstance(loss_res, dict):
                print('You have not use our provided loss func, please overwrite method train_detector_one_epoch')
                pass
            else:
                loss = loss_res['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for key, val in loss_res.items():
                    if key not in loss_dict_vec.keys():
                        loss_dict_vec[key] = []
                    loss_dict_vec[key].append(val.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict
