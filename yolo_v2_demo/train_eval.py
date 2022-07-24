import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.V2 import *
from Tool.BaseTools import get_voc_data_loader
from yolo_v2_demo.utils.get_pretrained_darknet_19 import get_pretained_dark_net_19
from yolo_v2_demo.utils.model_define import YOLOV2Net, DarkNet19
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WarmUpOptimizer:
    def __init__(
            self,
            paras,
            base_lr: float = 1e-3,
            warm_up_epoch: int = 1,
    ):
        self.optimizer = torch.optim.SGD(
            paras,
            base_lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        self.warm_up_epoch = warm_up_epoch
        self.base_lr = base_lr
        self.tmp_lr = base_lr

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warm(self,
             now_epoch_ind,
             max_epoch_size,
             now_batch_ind
             ):
        if now_epoch_ind < self.warm_up_epoch:
            tmp_lr = self.base_lr * pow((now_batch_ind + now_epoch_ind * max_epoch_size) * 1. / (self.warm_up_epoch * max_epoch_size), 4)
            self.set_lr(tmp_lr)

        elif now_epoch_ind == self.warm_up_epoch and now_batch_ind == 0:
            tmp_lr = self.base_lr
            self.set_lr(tmp_lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


class Helper:
    def __init__(
            self,
            model: nn.Module,
            opt_data_set: YOLOV2DataSetConfig,
            opt_trainer: YOLOV2TrainerConfig,
    ):
        self.detector = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.backbone = model.backbone  # type: nn.Module

        # be careful, darknet19 is not the detector

        self.opt_data_set = opt_data_set
        self.opt_trainer = opt_trainer

        self.trainer = YOLOV2Trainer(
            model,
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.image_size,
            self.opt_data_set.grid_number,
            self.opt_data_set.kinds_name,
            self.opt_trainer.iou_th
        )

        self.predictor = YOLOV2Predictor(
            self.opt_trainer.iou_th,
            self.opt_trainer.prob_th,
            self.opt_trainer.conf_th,
            self.opt_trainer.score_th,
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.kinds_name,
            self.opt_data_set.image_size,
            self.opt_data_set.grid_number
        )

        self.visualizer = YOLOV2Visualizer(
            model,
            self.predictor,
            self.opt_data_set.class_colors
        )

        self.evaluator = YOLOV2Evaluator(
            model,
            self.predictor
        )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV2Loss(
            self.opt_data_set.pre_anchor_w_h,
            self.opt_trainer.weight_position,
            self.opt_trainer.weight_conf_has_obj,
            self.opt_trainer.weight_conf_no_obj,
            self.opt_trainer.weight_cls_prob,
            self.opt_trainer.weight_iou_loss,
            self.opt_data_set.grid_number,
            image_size=self.opt_data_set.image_size,
            iou_th=self.opt_trainer.iou_th,
            loss_type=LOSS_TYPE
        )
        # already trained dark_net 19
        # so just train detector
        # optimizer = torch.optim.Adam(
        #     self.detector.parameters(),
        #     lr=self.opt_trainer.lr
        # )
        optimizer = torch.optim.SGD(
            self.detector.parameters(),
            lr=self.opt_trainer.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        for epoch in tqdm(range(self.opt_trainer.max_epoch_on_detector),
                          desc='training detector',
                          position=0):

            loss_dict = self.trainer.train_detector_one_epoch(
                data_loader_train,
                loss_func,
                optimizer,
                desc='train for detector epoch --> {}'.format(epoch)
            )

            print_info = '\n\n epoch: {}, loss info-->\n'.format(epoch)
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if True:
                # save model
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/model_pth_detector/'
                os.makedirs(saved_dir, exist_ok=True)
                torch.save(self.detector.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

                # show predict
                saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
                os.makedirs(saved_dir, exist_ok=True)

                self.visualizer.show_detect_results(
                    data_loader_test,
                    saved_dir
                )

                # eval mAP
                self.evaluator.eval_detector_mAP(
                    data_loader_test
                )


if __name__ == '__main__':
    GPU_ID = 1
    LOSS_TYPE = 1
    YOLOV2Predictor.TYPE = LOSS_TYPE
    YOLOV2Tools.TYPE = LOSS_TYPE
    fast_load = False

    trainer_opt = YOLOV2TrainerConfig()
    data_opt = YOLOV2DataSetConfig()
    trainer_opt.device = 'cuda:{}'.format(GPU_ID)
    trainer_opt.lr = 1e-4
    # dark_net_19 = get_pretained_dark_net_19(
    #     '/home/dell/PycharmProjects/YOLO/pre_trained/darknet19_72.96.pth'
    # )
    # dark_net_19 = DarkNet19()
    # net = YOLOV2Net(dark_net_19)

    from yolo_v2_demo.utils.get_yov2_resnet50 import YOLOV2Net
    net = YOLOV2Net(
        pretrained=True,
        num_anchors=len(data_opt.pre_anchor_w_h),
        num_classes=len(data_opt.kinds_name)
    )

    net.to(trainer_opt.device)
    helper = Helper(
        net,
        data_opt,
        trainer_opt
    )
    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    voc_train_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2007', '2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=True,
        mean=mean,
        std=std
    )
    voc_test_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=False,
        mean=mean,
        std=std
    )
    # helper.detector.load_state_dict(
    #     torch.load('/home/dell/data2/models/home/dell/PycharmProjects/YOLO/yolo_v2_demo/model_pth_detector/30.pth')
    # )
    helper.go(voc_train_loader, voc_test_loader)
