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
        self.detector.cuda()
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
        loss_dict_vec = {
            'position_loss': [],
            'has_obj_conf_loss': [],
            'no_obj_conf_loss': [],
            'cls_prob_loss': [],
            'total_loss': [],
        }
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            self.detector.train()
            images = images.cuda()
            targets = self.make_targets(labels, need_abs=True).cuda()
            output = self.detector(images)
            loss_tuple = yolo_loss_func(output, targets)
            if isinstance(loss_tuple, torch.Tensor) or len(loss_tuple) != 5:
                print('You have not use our provided loss func, please overwrite method train_detector_one_epoch')
                loss = loss_tuple if isinstance(loss_tuple, torch.Tensor) else loss_tuple[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                loss, position_loss, has_obj_conf_loss, no_obj_conf_loss, cls_prob_loss = loss_tuple
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_dict_vec['position_loss'].append(position_loss.item())
                loss_dict_vec['has_obj_conf_loss'].append(has_obj_conf_loss.item())
                loss_dict_vec['no_obj_conf_loss'].append(no_obj_conf_loss.item())
                loss_dict_vec['cls_prob_loss'].append(cls_prob_loss.item())
                loss_dict_vec['total_loss'].append(loss.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict


