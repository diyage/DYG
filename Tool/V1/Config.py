import numpy as np


class YOLOV1DataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    year: str = '2012'
    # data set root dir
    image_size: tuple = (448, 448)
    grid_number: tuple = (7, 7)
    kinds_name: list = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane',
                        'bicycle', 'boat', 'chair', 'diningtable',
                        'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']
    class_colors: list = [
        (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(len(kinds_name))
    ]
    pre_anchor_w_h: tuple = (
        (1.0, 1.0),
        (1.0, 1.0),
    )


class YOLOV1TrainerConfig:
    max_epoch_on_detector = 1000
    batch_size = 32
    lr: float = 1e-4
    conf_th: float = 0.5
    prob_th: float = 0.5
    score_th: float = 0.5
    iou_th: float = 0.6
    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 1.0
    weight_conf_has_obj: float = 5.0
    weight_conf_no_obj: float = 1.0
    weight_cls_prob: float = 1.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
