import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .BaseTools import YOLOV2Tools
from .Loss import YOLOV2Loss


class YOLOV2Trainer:
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

        self.dark_net = model.darknet19  # type: nn.Module
        self.dark_net.cuda()
        # be careful, darknet19 is not the detector

        self.pre_anchor_w_h = pre_anchor_w_h
        self.image_size = image_size
        self.grid_number = grid_number
        self.kinds_name = kinds_name

    def make_targets(
            self,
            labels,
            need_abs: bool = False,
    ):
        return YOLOV2Tools.make_targets(
            labels,
            self.pre_anchor_w_h,
            self.image_size,
            self.grid_number,
            self.kinds_name,
            need_abs
        )

    def train_classifier_one_epoch(
            self,
            data_loader_train: DataLoader,
            ce_loss_func,
            optimizer: torch.optim.Optimizer,
            desc: str = ''
    ):
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            self.dark_net.train()
            images = images.cuda()
            labels = labels.cuda()

            output = self.dark_net(images)  # type: torch.Tensor
            loss = ce_loss_func(output, labels)  # type: torch.Tensor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_v2_loss_func: YOLOV2Loss,
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
            loss, position_loss, has_obj_conf_loss, no_obj_conf_loss, cls_prob_loss = yolo_v2_loss_func(output, targets)
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
            loss_dict[key] = sum(val)/len(val)
        return loss_dict

