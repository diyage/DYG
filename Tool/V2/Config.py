'''
COCO:
(0.57273, 0.677385),
(1.87446, 2.06253),
(3.33843, 5.47434),
(7.88282, 3.52778),
(9.77052, 9.16828)
VOC:
(1.3221, 1.73145),
(3.19275, 4.00944),
(5.05587, 8.09892),
(9.47112, 4.84053),
(11.2364, 10.0071)

'''

import numpy as np


class DataSetConfig:
    root_path: str = '/home/dell/下载/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
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
        (1.3221, 1.73145),
        (3.19275, 4.00944),
        (5.05587, 8.09892),
        (9.47112, 4.84053),
        (11.2364, 10.0071)
    )


class TrainConfig:
    max_epoch_on_image_net_224 = 160
    max_epoch_on_image_net_448 = 10
    max_epoch_on_detector = 1000

    batch_size = 32
    lr: float = 1e-3
    conf_th: float = 0.6
    prob_th: float = 0.6
    score_th: float = 0.5
    iou_th: float = 0.6
    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 5.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 5.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
