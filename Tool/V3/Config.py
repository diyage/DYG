
import numpy as np


class YOLOV3DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    grid_number: dict = {
        'C3': (52, 52),
        'C4': (26, 26),
        'C5': (13, 13),
    }
    pre_anchor_w_h: dict = {
        'C3': ((32.64/416*52, 47.68/416*52),
               (50.24/416*52, 108.16/416*52),
               (126.72/416*52, 96.32/416*52)),

        'C4': ((78.40/416*26, 201.92/416*26),
               (178.24/416*26, 178.56/416*26),
               (129.60/416*26, 294.72/416*26)),

        'C5': ((331.84/416*13, 194.56/416*13),
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


class YOLOV3TrainerConfig:
    max_epoch_on_image_net_224 = 160
    max_epoch_on_image_net_448 = 10
    max_epoch_on_detector = 200

    device: str = 'cuda:0'
    batch_size = 32
    lr: float = 1e-3
    conf_th: float = 0.0
    prob_th: float = 0.0
    score_th: float = 0.1
    iou_for_nms: float = 0.6
    iou_for_make_target: float = 0.6
    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0
    weight_iou_loss: float = 0.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
