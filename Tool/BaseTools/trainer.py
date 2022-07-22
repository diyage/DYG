import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    def __init__(
            self,
            model: nn.Module,
            pre_anchor_w_h: tuple,
            image_size: tuple,
            grid_number: tuple,
            kinds_name: list,
    ):
        self.detector = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.image_size = image_size
        self.grid_number = grid_number
        self.kinds_name = kinds_name
        self.pre_anchor_w_h = pre_anchor_w_h

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ) -> torch.Tensor:
        pass

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_loss_func: nn.Module,
            optimizer: torch.optim.Optimizer,
            desc: str = '',
    ):
        loss_dict_vec = {}
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            self.detector.train()
            images = images.to(self.device)
            targets = self.make_targets(labels, need_abs=True).to(self.device)
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


