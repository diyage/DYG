from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.data import *
import tools

from utils.augmentations import SSDAugmentation
from utils.vocapi_evaluator import VOCAPIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv2 Detection')
    parser.add_argument('-v', '--version', default='yolov2',
                        help='yolov2')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                        default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str,
                        help='Gamma update for SGD')

    return parser.parse_args()


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


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # 是否使用cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 是否使用多尺度训练
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 416
    else:
        train_size = 416
        val_size = 416

    cfg = train_cfg
    # 构建dataset类和dataloader类
    if args.dataset == 'voc':
        # 加载voc数据集
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir,
                               transform=SSDAugmentation(train_size)
                               )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader类
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=detection_collate,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 构建我们的模型
    if args.version == 'yolov2':
        from models.yolov2 import YOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO

        yolo_net = YOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True,
                          anchor_size=anchor_size)
        print('Let us train yolov2 on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
    model.to(device).train()

    # 使用 tensorboard 可视化训练过程
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)

    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # # 构建训练优化器
    # base_lr = args.lr
    # tmp_lr = base_lr
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=args.lr,
    #                       momentum=args.momentum,
    #                       weight_decay=args.weight_decay
    #                       )
    optimizer = WarmUpOptimizer(
        model.parameters(),
        1e-3,
        warm_up_epoch=1,
    )

    max_epoch = cfg['max_epoch']  # 最大训练轮次
    epoch_size = len(dataset) // args.batch_size  # 每一训练轮次的迭代次数

    # 开始训练
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):

        # # 使用阶梯学习率衰减策略
        # if epoch in cfg['lr_epoch']:
        #     tmp_lr = tmp_lr * 0.1
        #     set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            optimizer.warm(epoch, epoch_size, iter_i)
            # 使用warm-up策略来调整早期的学习率
            # if not args.no_warm_up:
            #     if epoch < args.wp_epoch:
            #         tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
            #         set_lr(optimizer, tmp_lr)
            #
            #     elif epoch == args.wp_epoch and iter_i == 0:
            #         tmp_lr = base_lr
            #         set_lr(optimizer, tmp_lr)

            # 多尺度训练
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # 随机选择一个新的尺寸
                train_size = random.randint(10, 19) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # 插值
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            # 制作训练标签
            targets = [label.tolist() for label in targets]
            targets = tools.gt_creator(input_size=train_size,
                                       stride=yolo_net.stride,
                                       label_lists=targets,
                                       anchor_size=anchor_size
                                       )
            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # 前向推理和计算损失
            conf_loss, cls_loss, bbox_loss, iou_loss = model(images, target=targets)

            total_loss = conf_loss + cls_loss + bbox_loss  # + iou_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('obj loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('cls loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('box loss', bbox_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('iou loss', iou_loss.item(), iter_i + epoch * epoch_size)

                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || iou %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, max_epoch, iter_i, epoch_size, optimizer.tmp_lr,
                         conf_loss.item(),
                         cls_loss.item(),
                         bbox_loss.item(),
                         iou_loss.item(),
                         total_loss.item(),
                         train_size, t1 - t0),
                      flush=True)

                t0 = time.time()

        # evaluation
        if True:
            model.trainable = False
            model.set_grid(val_size)
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            # convert to training mode.
            model.trainable = True
            model.set_grid(train_size)
            model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            print('Saving state, epoch:', epoch + 1)
            torch.save(model.state_dict(), os.path.join(path_to_save,
                                                        args.version + '_' + repr(epoch + 1) + '.pth')
                       )


# def set_lr(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


if __name__ == '__main__':
    train()