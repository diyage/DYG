'''
COCO:
[[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

VOC:
[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

'''

import numpy as np


class YOLOV2DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    year: str = '2012'
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    grid_number: tuple = (13, 13)
    kinds_name: list = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane',
                        'bicycle', 'boat', 'chair', 'diningtable',
                        'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']

    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]

    pre_anchor_w_h: tuple = (
        (1.19, 1.98),
        (2.79, 4.59),
        (4.53, 8.92),
        (8.06, 5.29),
        (10.32, 10.65)
    )


class YOLOV2TrainerConfig:
    max_epoch_on_image_net_224 = 160
    max_epoch_on_image_net_448 = 10
    max_epoch_on_detector = 1000

    batch_size = 32
    lr: float = 1e-4
    conf_th: float = 0.0
    prob_th: float = 0.0
    score_th: float = 0.1
    iou_th: float = 0.6
    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
