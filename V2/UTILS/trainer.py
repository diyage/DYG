import torch
import torch.nn as nn
from V2.UTILS.model_define import YOLOV2Net
from torch.utils.data import DataLoader
from tqdm import tqdm
from V2.UTILS.config_define import *
import os
from V2.UTILS.others import YOLOV2Tools
from V2.UTILS.loss_define import YOLOV2Loss
from UTILS import CV2
import numpy as np
from typing import Union


class YOLOV2Trainer:
    def __init__(
            self,
            model: YOLOV2Net,
            opt_data_set: DataSetConfig,
            opt_trainer: TrainConfig,
    ):
        self.detector = model  # type: YOLOV2Net
        self.detector.cuda()
        # self.detector = nn.DataParallel(self.detector, device_ids=[0, 1])

        self.dark_net = model.darknet19
        self.dark_net.cuda()
        # self.dark_net = nn.DataParallel(self.dark_net, device_ids=[0, 1])

        self.opt_data_set = opt_data_set
        self.opt_trainer = opt_trainer

    def make_targets(
            self,
            labels
    ):
        return YOLOV2Tools.make_targets(labels,
                                        self.opt_data_set.pre_anchor_w_h,
                                        self.opt_data_set.image_size,
                                        self.opt_data_set.grid_number,
                                        self.opt_data_set.kinds_name)

    def decode_out(
            self,
            out_put: torch.Tensor,
            use_score: bool = True,
            use_conf: bool = False,
            out_is_target: bool = False,
    ) -> list:
        # 1 * _ * H * W or _ * H * W
        return YOLOV2Tools.decode_out(
            out_put,
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.image_size,
            self.opt_data_set.grid_number,
            self.opt_data_set.kinds_name,
            iou_th=self.opt_trainer.iou_th,
            conf_th=self.opt_trainer.conf_th,
            prob_th=self.opt_trainer.prob_th,
            use_score=use_score,
            use_conf=use_conf,
            out_is_target=out_is_target,
        )

    def __train_classifier_one_epoch(
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

    def __eval_classifier(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval classifier'
    ):
        vec = []
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.dark_net.eval()
            images = images.cuda()
            labels = labels.cuda()

            output = self.dark_net(images)  # type: torch.Tensor
            acc = (output.argmax(dim=-1) == labels).float().mean()
            vec.append(acc)

        accuracy = sum(vec) / len(vec)
        print('Acc: {:.3%}'.format(accuracy.item()))

    def train_on_image_net_224(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):

        ce_loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.dark_net.parameters(), lr=1e-4)

        for epoch in tqdm(range(self.opt_trainer.max_epoch_on_image_net_224),
                          desc='training on image_net_224',
                          position=0):

            self.__train_classifier_one_epoch(data_loader_train,
                                              ce_loss_func,
                                              optimizer,
                                              desc='train on image_net_224 epoch --> {}'.format(epoch))

            if epoch % 10 == 0:
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/model_pth_224/'
                os.makedirs(saved_dir, exist_ok=True)
                torch.save(self.dark_net.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))
                # eval image
                self.__eval_classifier(data_loader_test,
                                       desc='eval on image_net_224')

    def train_on_image_net_448(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        ce_loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.dark_net.parameters(), lr=1e-4)

        for epoch in tqdm(range(self.opt_trainer.max_epoch_on_image_net_448),
                          desc='training on image_net_448',
                          position=0):

            self.__train_classifier_one_epoch(data_loader_train,
                                              ce_loss_func,
                                              optimizer,
                                              desc='train on image_net_448 epoch --> {}'.format(epoch))

            if epoch % 10 == 0:
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/model_pth_448/'
                os.makedirs(saved_dir, exist_ok=True)
                torch.save(self.dark_net.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))
                # eval image
                self.__eval_classifier(data_loader_test,
                                       desc='eval on image_net_448')

    def __train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            yolo_v2_loss_func: YOLOV2Loss,
            optimizer: torch.optim.Optimizer,
            desc: str = '',
    ):
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_train,
                                                         desc=desc,
                                                         position=0)):
            self.detector.train()
            images = images.cuda()
            targets = self.make_targets(labels).cuda()
            output = self.detector(images)
            loss = yolo_v2_loss_func(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def __eval_detector(
            self,
            data_loader_test: DataLoader,
            desc: str = 'eval detector',
    ):
        # compute mAP
        gt_name_abs_pos_conf_vec = []
        pred_name_abs_pos_conf_vec = []
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.detector.eval()
            images = images.cuda()
            targets = self.make_targets(labels).cuda()
            output = self.detector(images)

            for image_index in range(images.shape[0]):
                gt_name_abs_pos_conf_vec.append(
                    self.decode_out(targets[image_index], use_score=False, use_conf=True, out_is_target=True)
                )
                pred_name_abs_pos_conf_vec.append(
                    self.decode_out(output[image_index], use_score=False, use_conf=True)
                )
        mAP = YOLOV2Tools.compute_mAP(
            pred_name_abs_pos_conf_vec,
            gt_name_abs_pos_conf_vec,
            kinds_name=self.opt_data_set.kinds_name,
            iou_th=self.opt_trainer.iou_th
        )
        print('mAP:{:.3%}'.format(mAP))

    def __show_detect_answer(
            self,
            data_loader_test: DataLoader,
            saved_dir: str,
            desc: str = 'show predict result'
    ):
        for batch_id, (images, labels) in enumerate(tqdm(data_loader_test,
                                                         desc=desc,
                                                         position=0)):
            self.detector.eval()
            images = images.cuda()
            targets = self.make_targets(labels).cuda()
            output = self.detector(images)
            for image_index in range(images.shape[0]):

                YOLOV2Tools.visualize(
                    images[image_index],
                    self.decode_out(targets[image_index], out_is_target=True),
                    saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, image_index),
                )

                YOLOV2Tools.visualize(
                    images[image_index],
                    self.decode_out(output[image_index]),
                    saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, image_index),
                )
            break

    def train_object_detector(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV2Loss(
            self.opt_data_set.pre_anchor_w_h,
            self.opt_trainer.weight_position,
            self.opt_trainer.weight_conf_has_obj,
            self.opt_trainer.weight_conf_no_obj,
            self.opt_trainer.weight_score
        )
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=1e-4)

        for epoch in tqdm(range(self.opt_trainer.max_epoch_on_detector),
                          desc='training detector',
                          position=0):

            self.__train_detector_one_epoch(data_loader_train,
                                            loss_func,
                                            optimizer,
                                            desc='train for detector epoch --> {}'.format(epoch))

            if epoch % 10 == 0:
                # save model
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/model_pth_detector/'
                os.makedirs(saved_dir, exist_ok=True)
                torch.save(self.detector.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))
                if epoch % 50 == 0:
                    # eval mAP
                    self.__eval_detector(data_loader_test)
                # show predict
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
                os.makedirs(saved_dir, exist_ok=True)
                self.__show_detect_answer(data_loader_test, saved_dir)
