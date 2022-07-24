import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.V2 import *
from Tool.BaseTools import get_voc_data_loader, WarmUpOptimizer
from yolo_v2_demo.utils.get_pretrained_darknet_19 import get_pretained_dark_net_19
from yolo_v2_demo.utils.model_define import YOLOV2Net, DarkNet19
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        sgd_optimizer = torch.optim.SGD(
            self.detector.parameters(),
            lr=self.opt_trainer.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        warm_optimizer = WarmUpOptimizer(
            sgd_optimizer,
            base_lr=self.opt_trainer.lr,
            warm_up_epoch=1
        )

        for epoch in tqdm(range(self.opt_trainer.max_epoch_on_detector),
                          desc='training detector',
                          position=0):

            if epoch % 50 == 0 and epoch != 0:
                warm_optimizer.set_lr(warm_optimizer.tmp_lr * 0.1)

            loss_dict = self.trainer.train_detector_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                now_epoch=epoch,
                desc='train for detector epoch --> {}'.format(epoch)
            )

            print_info = '\n\nepoch: {} [ now lr:{:.6f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
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
    GPU_ID = 0
    LOSS_TYPE = 0
    YOLOV2Predictor.TYPE = LOSS_TYPE
    YOLOV2Tools.TYPE = LOSS_TYPE

    trainer_opt = YOLOV2TrainerConfig()
    data_opt = YOLOV2DataSetConfig()
    trainer_opt.device = 'cuda:{}'.format(GPU_ID)

    dark_net_19 = get_pretained_dark_net_19(
        '/home/dell/PycharmProjects/YOLO/pre_trained/darknet19_72.96.pth'
    )
    # dark_net_19 = DarkNet19()
    net = YOLOV2Net(dark_net_19)

    net.to(trainer_opt.device)
    helper = Helper(
        net,
        data_opt,
        trainer_opt
    )

    voc_train_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2007', '2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=True,
        mean=data_opt.mean,
        std=data_opt.std
    )
    voc_test_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2012'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=False,
        mean=data_opt.mean,
        std=data_opt.std
    )
    # helper.detector.load_state_dict(
    #     torch.load('/home/dell/data2/models/home/dell/PycharmProjects/YOLO/yolo_v2_demo/model_pth_detector/30.pth')
    # )
    helper.go(voc_train_loader, voc_test_loader)
