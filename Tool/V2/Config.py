'''
COCO:
[[0.53, 0.79], [1.71, 2.36], [2.89, 6.44], [6.33, 3.79], [9.03, 9.74]]

VOC:
[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]

'''

import numpy as np


class YOLOV2DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    # grid_number: tuple = (13, 13)
    image_shrink_rate: tuple = (32, 32)
    kinds_name: list = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]

    # pre_anchor_w_h: tuple = (
    #     (1.19, 1.98),
    #     (2.79, 4.59),
    #     (4.53, 8.92),
    #     (8.06, 5.29),
    #     (10.32, 10.65)
    # )
    pre_anchor_w_h_rate: tuple = (
        (0.09153846153846154, 0.1523076923076923),
        (0.21461538461538462, 0.35307692307692307),
        (0.3484615384615385, 0.6861538461538461),
        (0.62, 0.40692307692307694),
        (0.7938461538461539, 0.8192307692307692)
    )

    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class YOLOV2TrainerConfig:
    max_epoch_on_detector = 200
    eval_frequency = 10
    num_workers: int = 4
    device: str = 'cuda:0'
    batch_size = 32
    lr: float = 1e-3
    warm_up_end_epoch: int = 1

    conf_th_for_show: float = 0.0
    prob_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    conf_th_for_eval: float = 0.0
    prob_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001

    iou_th_for_show: float = 0.5
    iou_th_for_eval: float = 0.5
    iou_th_for_make_target: float = 0.5

    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0
    weight_iou_loss: float = 0.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
