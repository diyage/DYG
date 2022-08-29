import torch
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from Tool.FCNDesNet101_SS import *
from Tool.BaseTools import WarmUpOptimizer
import albumentations as alb
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WarmUpCosineAnnealOptimizer(WarmUpOptimizer):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_epoch_for_train: int,
            base_lr: float = 1e-3,
            warm_up_end_epoch: int = 1,
    ):
        super().__init__(
            optimizer,
            base_lr,
            warm_up_end_epoch
        )
        self.max_epoch_for_train = max_epoch_for_train

    def warm(
            self,
            now_epoch_ind,
            now_batch_ind,
            max_batch_ind
    ):
        if now_epoch_ind < self.warm_up_epoch:
            self.tmp_lr = self.base_lr * pow(
                (now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
            self.set_lr(self.tmp_lr)
        else:
            T = (self.max_epoch_for_train - self.warm_up_epoch + 1) * max_batch_ind
            t = (now_epoch_ind - self.warm_up_epoch) * max_batch_ind + now_batch_ind

            lr = 1.0 / 2 * (1.0 + math.cos(t * math.pi / T)) * self.base_lr
            self.set_lr(lr)


class Helper:
    def __init__(
            self,
            model: FCNResnet101,
            opt: SSConfig,
            restore_epoch: int = -1
    ):
        self.detector = model  # type: FCNResnet101
        self.device = next(model.parameters()).device

        self.config = opt

        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = SSTrainer(
            model,
        )

        self.predictor_for_show = SSPredictor()

        self.predictor_for_eval = SSPredictor()

        self.visualizer = SSVisualizer(
            model,
            self.predictor_for_show,
            image_mean=self.config.data_config.mean,
            image_std=self.config.data_config.std,
        )

        self.my_evaluator = SSEvaluator(
            model,
            self.predictor_for_eval,
        )

    def restore(
            self,
            epoch: int
    ):
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.detector.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.detector.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.detector.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def show_detect_results(
            self,
            data_loader_test: DataLoader,
            epoch: int
    ):
        with torch.no_grad():
            saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.visualizer.show_detect_results(
                data_loader_test,
                saved_dir,
                desc='[show predict results]'
            )

    def eval_semantic_segmentation_accuracy(
            self,
            data_loader_test: DataLoader,
    ):
        with torch.no_grad():
            self.my_evaluator.eval_semantic_segmentation_accuracy(
                data_loader_test,
                desc='[eval semantic segmentation accuracy]'
            )

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):
        loss_func = FocalLoss()

        sgd_optimizer = torch.optim.SGD(
            self.detector.parameters(),
            lr=self.config.train_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )
        # adam_optimizer = torch.optim.Adam(
        #     self.detector.parameters(),
        #     lr=self.config.train_config.lr,
        # )

        warm_optimizer = WarmUpCosineAnnealOptimizer(
            sgd_optimizer,
            self.config.train_config.max_epoch_on_detector,
            base_lr=self.config.train_config.lr,
            warm_up_end_epoch=self.config.train_config.warm_up_end_epoch
        )

        for epoch in tqdm(range(self.restore_epoch + 1, self.config.train_config.max_epoch_on_detector),
                          desc='training detector',
                          position=0):

            loss_dict = self.trainer.train_detector_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                now_epoch=epoch,
                desc='[train for detector epoch: {}/{}]'.format(epoch, self.config.train_config.max_epoch_on_detector-1)
            )

            print_info = '\n\nepoch: {} [ now lr:{:.8f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_config.eval_frequency == 0:

                # save model
                self.save(epoch)

                # show predict
                self.show_detect_results(data_loader_test, epoch)

                # eval accuracy
                self.eval_semantic_segmentation_accuracy(data_loader_test)


if __name__ == '__main__':
    GPU_ID = 0
    """
        set config
    """
    config = SSConfig()
    config.train_config.device = 'cuda:{}'.format(GPU_ID)
    config.train_config.lr = 1e-4
    """
        build net
    """
    net = get_fcn_resnet101(pretrained=False, num_classes=21)

    net.to(config.train_config.device)
    """
        get data
    """
    trans_train = alb.Compose([
        alb.HueSaturationValue(),
        alb.Rotate(),
        alb.RandomBrightnessContrast(),
        alb.ColorJitter(),
        alb.Blur(),
        alb.GaussNoise(),
        alb.HorizontalFlip(),
        alb.VerticalFlip(),
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    trans_test = alb.Compose([
        alb.Resize(config.data_config.image_size[0], config.data_config.image_size[1]),
        alb.Normalize(
            mean=config.data_config.mean,
            std=config.data_config.std
        )
    ], bbox_params=alb.BboxParams(format='pascal_voc'))

    voc_train_loader = get_voc_for_all_tasks_loader(
        config.data_config.root_path,
        ['2007', '2012'],
        train=True,
        image_size=config.data_config.image_size[0],
        mean=config.data_config.mean,
        std=config.data_config.std,
        trans_form=trans_train,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_bbox=False,
        use_mask_type=-1
    )
    voc_test_loader = get_voc_for_all_tasks_loader(
        config.data_config.root_path,
        ['2007'],
        train=False,
        image_size=config.data_config.image_size[0],
        mean=config.data_config.mean,
        std=config.data_config.std,
        trans_form=trans_test,
        batch_size=config.train_config.batch_size,
        num_workers=config.train_config.num_workers,
        use_bbox=False,
        use_mask_type=-1
    )

    helper = Helper(
        net,
        config,
        restore_epoch=-1
    )
    helper.restore(50)
    # helper.eval_map()
    # helper.show_detect_results(voc_test_loader, 0)
    # helper.eval_semantic_segmentation_accuracy(voc_test_loader)
    helper.go(voc_train_loader, voc_test_loader)

