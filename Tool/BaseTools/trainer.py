import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union


class WarmUpOptimizer:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            base_lr: float = 1e-3,
            warm_up_epoch: int = 1,
    ):
        self.optimizer = optimizer
        self.set_lr(base_lr)

        self.warm_up_epoch = warm_up_epoch
        self.base_lr = base_lr
        self.tmp_lr = base_lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warm(self,
             now_epoch_ind,
             now_batch_ind,
             max_batch_ind
             ):
        if now_epoch_ind < self.warm_up_epoch:
            self.tmp_lr = self.base_lr * pow((now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
            self.set_lr(self.tmp_lr)

        elif now_epoch_ind == self.warm_up_epoch and now_batch_ind == 0:
            self.tmp_lr = self.base_lr
            self.set_lr(self.tmp_lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


class BaseTrainer:
    def __init__(
            self,
            model: nn.Module,
            pre_anchor_w_h: tuple,
            image_size: tuple,
            grid_number: tuple,
            kinds_name: list,
            iou_th: float = 0.6
    ):
        self.detector = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.image_size = image_size
        self.grid_number = grid_number
        self.kinds_name = kinds_name
        self.pre_anchor_w_h = pre_anchor_w_h
        self.iou_th = iou_th

    def make_targets(
            self,
            labels,
    ) -> torch.Tensor:
        pass

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_loss_func: nn.Module,
            optimizer: Union[torch.optim.Optimizer, WarmUpOptimizer],
            now_epoch: int,
            desc: str = '',
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)

        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            optimizer.warm(
                now_epoch,
                batch_id,
                max_batch_ind
            )

            self.detector.train()
            images = images.to(self.device)
            targets = self.make_targets(labels).to(self.device)
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


