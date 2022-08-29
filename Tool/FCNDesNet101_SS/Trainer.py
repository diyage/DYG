from Tool.FCNDesNet101_SS.Tools import SSTools
from Tool.FCNDesNet101_SS.Model import FCNResnet101
from Tool.FCNDesNet101_SS.Loss import FocalLoss
from Tool.BaseTools import WarmUpOptimizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from typing import Union


class SSTrainer:
    def __init__(
            self,
            model: FCNResnet101,
    ):
        self.detector = model
        self.device = next(model.parameters()).device

    def make_targets(
            self,
            labels,
    ):
        targets = SSTools.make_target(
            labels,
        )
        return targets.to(self.device)

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_loss_func: FocalLoss,
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

            targets = self.make_targets(masks)

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
