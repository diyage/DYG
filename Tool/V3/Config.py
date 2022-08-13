
import numpy as np


class YOLOV3DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    # grid_number: dict = {
    #     'C3': (52, 52),
    #     'C4': (26, 26),
    #     'C5': (13, 13),
    # }
    image_shrink_rate: dict = {
        'C3': (8, 8),
        'C4': (16, 16),
        'C5': (32, 32),
    }
    # pre_anchor_w_h: dict = {
    #     'C3': ((32.64/416*52, 47.68/416*52),
    #            (50.24/416*52, 108.16/416*52),
    #            (126.72/416*52, 96.32/416*52)),
    #
    #     'C4': ((78.40/416*26, 201.92/416*26),
    #            (178.24/416*26, 178.56/416*26),
    #            (129.60/416*26, 294.72/416*26)),
    #
    #     'C5': ((331.84/416*13, 194.56/416*13),
    #            (227.84/416*13, 325.76/416*13),
    #            (365.44/416*13, 358.72/416*13)),
    # }
    pre_anchor_w_h_rate: dict = {
        'C3': ((0.07846153846153846, 0.11461538461538462),
               (0.12076923076923077, 0.26),
               (0.3046153846153846, 0.23153846153846153)),

        'C4': ((0.18846153846153849, 0.48538461538461536),
               (0.4284615384615385, 0.42923076923076925),
               (0.31153846153846154, 0.7084615384615385)),

        'C5': ((0.7976923076923076, 0.4676923076923077),
               (0.5476923076923077, 0.783076923076923),
               (0.8784615384615384, 0.8623076923076923)),
    }
    single_an: int = 3
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

    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class YOLOV3TrainerConfig:
    max_epoch_on_detector = 200
    eval_frequency = 10
    num_workers: int = 4
    device: str = 'cuda:0'
    batch_size = 16
    lr: float = 1e-3
    warm_up_end_epoch: int = 5

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


class YOLOV3Config:
    data_config = YOLOV3DataSetConfig()
    train_config = YOLOV3TrainerConfig()


