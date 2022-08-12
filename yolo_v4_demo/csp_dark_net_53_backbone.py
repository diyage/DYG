import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.V4 import *
from Tool.BaseTools import WarmUpOptimizer, BaseTransform
from Tool.V4.DatasetDefine import get_stronger_voc_data_loader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Helper:
    def __init__(
            self,
            model: YOLOV4Model,
            opt: YOLOV4Config,
    ):
        self.detector = model  # type: YOLOV4Model
        self.device = next(model.parameters()).device
        self.backbone = model.backbone  # type: nn.Module

        self.config = opt

        self.trainer = YOLOV4Trainer(
            model,
            self.config.data_config.pre_anchor_w_h,
            self.config.data_config.image_size,
            self.config.data_config.grid_number,
            self.config.data_config.kinds_name,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

        self.predictor_for_show = YOLOV4Predictor(
            self.config.show_config.iou_th_for_show,
            self.config.show_config.prob_th_for_show,
            self.config.show_config.conf_th_for_show,
            self.config.show_config.score_th_for_show,
            self.config.data_config.pre_anchor_w_h,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.grid_number,
            self.config.data_config.single_an
        )

        self.predictor_for_eval = YOLOV4Predictor(
            self.config.eval_config.iou_th_for_eval,
            self.config.eval_config.prob_th_for_eval,
            self.config.eval_config.conf_th_for_eval,
            self.config.eval_config.score_th_for_eval,
            self.config.data_config.pre_anchor_w_h,
            self.config.data_config.kinds_name,
            self.config.data_config.image_size,
            self.config.data_config.grid_number,
            self.config.data_config.single_an
        )

        self.visualizer = YOLOV4Visualizer(
            model,
            self.predictor_for_show,
            self.config.data_config.class_colors,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

        self.formal_evaluator = YOLOV4FormalEvaluator(
            model,
            self.predictor_for_eval,
            self.config.data_config.root_path,
            self.config.data_config.image_size[0],
            self.config.train_config.device,
            transform=BaseTransform(self.config.data_config.image_size[0]),
            labelmap=self.config.data_config.kinds_name,
        )
        self.my_evaluator = YOLOV4Evaluator(
            model,
            self.predictor_for_eval,
            self.config.iou_th_for_make_target,
            multi_gt=self.config.multi_gt
        )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = YOLOV4Loss(
            self.config.data_config.pre_anchor_w_h,
            self.config.data_config.grid_number,
            self.config.data_config.single_an,
            self.config.train_config.weight_position,
            self.config.train_config.weight_conf_has_obj,
            self.config.train_config.weight_conf_no_obj,
            self.config.train_config.weight_cls_prob,
            image_size=self.config.data_config.image_size,
        )
        # already trained dark_net 53
        # so just train detector
        # optimizer = torch.optim.Adam(
        #     self.detector.parameters(),
        #     lr=self.opt_trainer.lr
        # )
        sgd_optimizer = torch.optim.SGD(
            self.detector.parameters(),
            lr=self.config.train_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        warm_optimizer = WarmUpOptimizer(
            sgd_optimizer,
            base_lr=self.config.train_config.lr,
            warm_up_epoch=self.config.train_config.warm_up_end_epoch
        )

        for epoch in tqdm(range(self.config.train_config.max_epoch_on_detector),
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

            if epoch % self.config.eval_config.eval_frequency == 0:
                # save model
                saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
                os.makedirs(saved_dir, exist_ok=True)
                torch.save(self.detector.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

                # show predict
                saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
                self.visualizer.show_detect_results(
                    data_loader_test,
                    saved_dir
                )

                # eval mAP
                self.formal_evaluator.eval_detector_mAP()
                # self.my_evaluator.eval_detector_mAP(data_loader_test)


if __name__ == '__main__':
    GPU_ID = 0

    config = YOLOV4Config()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)

    csp_dark_net_53 = get_backbone_csp_darknet_53(
        '/home/dell/PycharmProjects/YOLO/pre_trained/cspdarknet53.pth'
    )

    net = YOLOV4Model(
        csp_dark_net_53,
        config.data_config.single_an,
        num_classes=len(config.data_config.kinds_name)
    )

    net.to(config.train_config.device)
    helper = Helper(
        net,
        config
    )

    voc_train_loader = get_stronger_voc_data_loader(
        config.data_config.root_path,
        ['2007', '2012'],
        image_size=config.data_config.image_size,
        batch_size=config.train_config.batch_size,
        train=True,
        mean=config.data_config.mean,
        std=config.data_config.std,
        use_mosaic=config.train_config.use_mosaic,
        use_mixup=config.train_config.use_mixup
    )
    voc_test_loader = get_stronger_voc_data_loader(
        config.data_config.root_path,
        ['2007'],
        image_size=config.data_config.image_size,
        batch_size=config.train_config.batch_size,
        train=False,
        mean=config.data_config.mean,
        std=config.data_config.std,
        use_mosaic=False,
        use_mixup=False,
    )
    # helper.detector.load_state_dict(
    #     torch.load('/home/dell/data2/models/home/dell/PycharmProjects/YOLO/yolo_v4_demo/model_pth_detector/50.pth')
    # )
    # helper.my_evaluator.eval_detector_mAP(voc_test_loader)
    # helper.formal_evaluator.eval_detector_mAP()
    helper.go(voc_train_loader, voc_test_loader)
