
class DataSetConfig:
    root_path: str = '/home/dell/下载/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    image_net_dir: str = '/home/dell/data/DataSet/imageNet/data/'
    # data set root dir
    image_size: tuple = (416, 416)
    grid_number: tuple = (13, 13)
    kinds_name: list = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane',
                        'bicycle', 'boat', 'chair', 'diningtable',
                        'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']

    pre_anchor_w_h: tuple = ([1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65])


class TrainConfig:
    max_epoch_on_image_net_224 = 160
    max_epoch_on_image_net_448 = 10
    max_epoch_on_detector = 200

    batch_size = 32
    lr: float = 1e-4
    conf_th: float = 0.1
    prob_th: float = 0.1
    iou_th: float = 0.5
    ABS_PATH: str = '/home/dell/data2/models/'
    weight_position: float = 1.0
    weight_conf_has_obj: float = 1.0
    weight_conf_no_obj: float = 1.0
    weight_score: float = 1.0
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
