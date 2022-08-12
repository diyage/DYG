
import numpy as np


class YOLOV4DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    grid_number: dict = {
        'for_s': (52, 52),
        'for_m': (26, 26),
        'for_l': (13, 13),
    }
    pre_anchor_w_h: dict = {
        'for_s': ((32.64/416*52, 47.68/416*52),
                  (50.24/416*52, 108.16/416*52),
                  (126.72/416*52, 96.32/416*52)),

        'for_m': ((78.40/416*26, 201.92/416*26),
                  (178.24/416*26, 178.56/416*26),
                  (129.60/416*26, 294.72/416*26)),

        'for_l': ((331.84/416*13, 194.56/416*13),
                  (227.84/416*13, 325.76/416*13),
                  (365.44/416*13, 358.72/416*13)),
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


class YOLOV4TrainConfig:
    max_epoch_on_detector = 200

    device: str = 'cuda:0'
    batch_size = 16
    lr: float = 1e-3
    warm_up_end_epoch: int = 5
    use_mosaic: bool = True
    use_mixup: bool = True

    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0
    weight_iou_loss: float = 0.0


class YOLOV4EvalConfig:

    eval_frequency = 10

    conf_th_for_eval: float = 0.0
    prob_th_for_eval: float = 0.0
    score_th_for_eval: float = 0.001
    iou_th_for_eval: float = 0.5


class YOLOV4VisualizationConfig:

    conf_th_for_show: float = 0.0
    prob_th_for_show: float = 0.0
    score_th_for_show: float = 0.3

    iou_th_for_show: float = 0.5


class YOLOV4Config:
    ABS_PATH: str = '/home/dell/data2/models/'
    iou_th_for_make_target: float = 0.5
    multi_gt: bool = True
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = YOLOV4DataSetConfig()
    train_config = YOLOV4TrainConfig()
    eval_config = YOLOV4EvalConfig()
    show_config = YOLOV4VisualizationConfig()


