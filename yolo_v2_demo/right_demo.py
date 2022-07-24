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
from Tool.BaseTools import WarmUpOptimizer
from Tool.BaseTools import BaseEvaluator
from Tool.V2 import YOLOV2Tools
import torch
import torch.nn as nn


class MyEvaluator(BaseEvaluator):
    def __init__(
            self,
            model,
            kinds_name: list
    ):
        super().__init__()
        self.detector = model  # type: nn.Module
        self.backbone = model.backbone  # type: nn.Module
        self.device = next(model.parameters()).device
        self.kinds_name = kinds_name

    def eval_detector_mAP(
            self,
            data_loader_test,
            kinds_name: list,
            iou_th: float,

    ):

        pre_decode = []
        gt_decode = []
        for iter_i, (images, targets) in enumerate(data_loader_test):

            targets = [label.tolist() for label in targets]

            images = images.to(self.device)

            for image_index in range(images.shape[0]):
                ##################################################
                gt_kps_s = []
                for gt_label in targets[image_index]:
                    # get a bbox coords
                    cls_ind = int(gt_label[-1])
                    bbox = gt_label[:-1]
                    score = 1.0
                    kps = [
                        kinds_name[cls_ind],
                        (bbox[0] * 416, bbox[1] * 416, bbox[2] * 416, bbox[3] * 416,),
                        score
                    ]
                    gt_kps_s.append(kps)

                gt_decode.append(gt_kps_s)
                ##################################################

                ##################################################
                pre_kps_s = []
                bboxes, scores, cls_inds = self.detector(images[image_index].unsqueeze(0))
                for i in range(len(cls_inds)):
                    cls_ind = cls_inds[i]
                    bbox = bboxes[i]
                    score = scores[i]

                    kps = [
                        kinds_name[cls_ind],
                        (bbox[0]*416, bbox[1]*416, bbox[2]*416, bbox[3]*416,),
                        score
                    ]
                    pre_kps_s.append(kps)
                pre_decode.append(pre_kps_s)
                ##################################################

        # compute mAP
        assert len(pre_decode) == len(gt_decode)
        record = {
            key: [[], [], 0] for key in kinds_name
            # kind_name: [tp_list, score_list, gt_num]
        }

        for image_index in range(len(gt_decode)):

            res = YOLOV2Tools.get_pre_kind_name_tp_score_and_gt_num(
                pre_decode[image_index],
                gt_decode[image_index],
                kinds_name=kinds_name,
                iou_th=iou_th,
            )

            for pre_kind_name, is_tp, pre_score in res[0]:
                record[pre_kind_name][0].append(is_tp)  # tp list
                record[pre_kind_name][1].append(pre_score)  # score list

            for kind_name, gt_num in res[1].items():
                record[kind_name][2] += gt_num

        #
        ap_vec = []
        for kind_name in kinds_name:
            tp_list, score_list, gt_num = record[kind_name]
            recall, precision = YOLOV2Tools.calculate_pr(gt_num, tp_list, score_list)
            kind_name_ap = YOLOV2Tools.voc_ap(recall, precision)
            ap_vec.append(kind_name_ap)

        mAP = np.mean(ap_vec)
        print('\nmAP:{:.2%}'.format(mAP))


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
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay
                          )
    optimizer = WarmUpOptimizer(
        optimizer,
        1e-3,
        warm_up_epoch=1,
    )
    my_evaluator = MyEvaluator(
        model,
        VOC_CLASSES
    )
    max_epoch = cfg['max_epoch']  # 最大训练轮次
    epoch_size = len(dataset) // args.batch_size  # 每一训练轮次的迭代次数
    print(epoch_size)
    print(len(dataloader))
    # 开始训练
    t0 = time.time()
    for epoch in range(args.start_epoch, max_epoch):

        # # 使用阶梯学习率衰减策略
        # if epoch in cfg['lr_epoch']:
        #     tmp_lr = tmp_lr * 0.1
        #     set_lr(optimizer, tmp_lr)

        for iter_i, (images, targets) in enumerate(dataloader):
            optimizer.warm(epoch, iter_i, epoch_size)
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

            # # evaluate
            # evaluator.evaluate(model)

            from torch.utils.data import DataLoader
            dl = DataLoader(
                evaluator.dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=detection_collate
            )
            my_evaluator.eval_detector_mAP(
                dl,
                kinds_name=VOC_CLASSES,
                iou_th=0.6,
            )

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