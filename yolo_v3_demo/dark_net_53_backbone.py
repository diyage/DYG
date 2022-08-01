import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.V3 import *
from Tool.BaseTools import get_voc_data_loader, WarmUpOptimizer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Helper:
    def __init__(
            self,
            model: nn.Module,
            opt_data_set: YOLOV3DataSetConfig,
            opt_trainer: YOLOV3TrainerConfig,
    ):
        self.detector = model  # type: nn.Module
        self.device = next(model.parameters()).device
        self.backbone = model.backbone  # type: nn.Module

        self.opt_data_set = opt_data_set
        self.opt_trainer = opt_trainer

        self.trainer = YOLOV3Trainer(
            model,
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.image_size,
            self.opt_data_set.grid_number,
            self.opt_data_set.kinds_name,
            self.opt_trainer.iou_for_make_target,
        )

        self.predictor = YOLOV3Predictor(
            self.opt_trainer.iou_for_nms,
            self.opt_trainer.prob_th,
            self.opt_trainer.conf_th,
            self.opt_trainer.score_th,
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.kinds_name,
            self.opt_data_set.image_size,
            self.opt_data_set.grid_number,
            self.opt_data_set.single_an
        )

        self.visualizer = YOLOV3Visualizer(
            model,
            self.predictor,
            self.opt_data_set.class_colors,
        )

        self.evaluator = FormalEvaluator(
            self.predictor,
            self.opt_data_set.root_path,
            self.opt_data_set.image_size[0],
            self.opt_trainer.device,
            transform=BaseTransform(self.opt_data_set.image_size[0]),
            labelmap=self.opt_data_set.kinds_name,
        )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV3Loss(
            self.opt_data_set.pre_anchor_w_h,
            self.opt_data_set.grid_number,
            self.opt_data_set.single_an,
            self.opt_trainer.weight_position,
            self.opt_trainer.weight_conf_has_obj,
            self.opt_trainer.weight_conf_no_obj,
            self.opt_trainer.weight_cls_prob,
            image_size=self.opt_data_set.image_size,
        )
        # already trained dark_net 53
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
            warm_up_epoch=self.opt_trainer.warm_up_end_epoch
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

            if epoch % self.opt_trainer.eval_frequency == 0:
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
                self.evaluator.evaluate(self.detector)


if __name__ == '__main__':
    GPU_ID = 1

    trainer_opt = YOLOV3TrainerConfig()
    data_opt = YOLOV3DataSetConfig()
    trainer_opt.device = 'cuda:{}'.format(GPU_ID)
    trainer_opt.lr = 1e-3
    trainer_opt.batch_size = 8
    trainer_opt.warm_up_end_epoch = 2
    trainer_opt.eval_frequency = 1

    net = YOLOV3Net(
        '/home/dell/PycharmProjects/YOLO/pre_trained/darknet53_75.42.pth',
        data_opt.single_an,
        num_classes=len(data_opt.kinds_name)
    )

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
        ['2007'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=False,
        mean=data_opt.mean,
        std=data_opt.std
    )
    helper.go(voc_train_loader, voc_test_loader)
