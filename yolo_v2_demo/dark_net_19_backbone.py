import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.V2 import *
from Tool.BaseTools import get_voc_data_loader, WarmUpOptimizer, BaseTransform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Helper:
    def __init__(
            self,
            model: YOLOV2Model,
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
            self.opt_data_set.pre_anchor_w_h_rate,
            self.opt_data_set.image_size,
            self.opt_data_set.image_shrink_rate,
            self.opt_data_set.kinds_name,
            self.opt_trainer.iou_th_for_make_target
        )

        self.predictor_for_show = YOLOV2Predictor(
            self.opt_trainer.iou_th_for_show,
            self.opt_trainer.prob_th_for_show,
            self.opt_trainer.conf_th_for_show,
            self.opt_trainer.score_th_for_show,
            self.opt_data_set.pre_anchor_w_h_rate,
            self.opt_data_set.kinds_name,
            self.opt_data_set.image_size,
            self.opt_data_set.image_shrink_rate
        )

        self.predictor_for_eval = YOLOV2Predictor(
            self.opt_trainer.iou_th_for_eval,
            self.opt_trainer.prob_th_for_eval,
            self.opt_trainer.conf_th_for_eval,
            self.opt_trainer.score_th_for_eval,
            self.opt_data_set.pre_anchor_w_h_rate,
            self.opt_data_set.kinds_name,
            self.opt_data_set.image_size,
            self.opt_data_set.image_shrink_rate
        )

        self.visualizer = YOLOV2Visualizer(
            model,
            self.predictor_for_show,
            self.opt_data_set.class_colors,
            self.opt_trainer.iou_th_for_make_target,
        )

        self.formal_evaluator = YOLOV2FormalEvaluator(
            model,
            self.predictor_for_eval,
            self.opt_data_set.root_path,
            self.opt_data_set.image_size[0],
            self.opt_trainer.device,
            transform=BaseTransform(self.opt_data_set.image_size[0]),
            labelmap=self.opt_data_set.kinds_name,
        )

        self.my_evaluator = YOLOV2Evaluator(
            model,
            self.predictor_for_eval,
            self.opt_trainer.iou_th_for_make_target
        )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV2Loss(
            self.opt_data_set.pre_anchor_w_h_rate,
            self.opt_trainer.weight_position,
            self.opt_trainer.weight_conf_has_obj,
            self.opt_trainer.weight_conf_no_obj,
            self.opt_trainer.weight_cls_prob,
            self.opt_trainer.weight_iou_loss,
            self.opt_data_set.image_shrink_rate,
            image_size=self.opt_data_set.image_size,
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

                with torch.no_grad():
                    # show predict
                    saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
                    self.visualizer.show_detect_results(
                        data_loader_test,
                        saved_dir
                    )

                    # eval mAP
                    self.formal_evaluator.eval_detector_mAP()
                    # self.my_evaluator.eval_detector_mAP(
                    #     data_loader_test
                    # )


if __name__ == '__main__':
    GPU_ID = 1

    # LOSS_TYPE = 1
    # YOLOV2Tools.TYPE = LOSS_TYPE
    # # mAP is about 0.73~0.74

    LOSS_TYPE = 0
    YOLOV2Tools.TYPE = LOSS_TYPE
    # mAP is about 0.73~0.74

    trainer_opt = YOLOV2TrainerConfig()
    data_opt = YOLOV2DataSetConfig()
    trainer_opt.device = 'cuda:{}'.format(GPU_ID)

    dark_net_19 = get_backbone_dark_net_19(
        '/home/dell/PycharmProjects/YOLO/pre_trained/darknet19_72.96.pth'
    )
    net = YOLOV2Model(dark_net_19)

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
        num_workers=trainer_opt.num_workers,
        mean=data_opt.mean,
        std=data_opt.std
    )
    voc_test_loader = get_voc_data_loader(
        data_opt.root_path,
        ['2007'],
        data_opt.image_size,
        trainer_opt.batch_size,
        train=False,
        num_workers=trainer_opt.num_workers,
        mean=data_opt.mean,
        std=data_opt.std
    )
    # helper.detector.load_state_dict(
    #     torch.load('/home/dell/data2/models/home/dell/PycharmProjects/YOLO/yolo_v2_demo/model_pth_detector/30.pth')
    # )

    # helper.my_evaluator.eval_detector_mAP(voc_test_loader)
    # helper.formal_evaluator.eval_detector_mAP()
    helper.go(voc_train_loader, voc_test_loader)
