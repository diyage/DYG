
class DataSetConfig:
    root_path: str = '/home/dell/下载/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
    # data set root dir
    image_size: tuple = (448, 448)
    grid_number: tuple = (7, 7)
    kinds_name: list = ['horse', 'person', 'bottle', 'dog', 'tvmonitor', 'car', 'aeroplane',
                        'bicycle', 'boat', 'chair', 'diningtable',
                        'pottedplant', 'train', 'cat', 'sofa', 'bird', 'sheep', 'motorbike', 'bus', 'cow']


class TrainConfig:
    max_epoch = 500
    batch_size = 128
    lr: float = 1e-4
    conf_th: float = 0.1
    prob_th: float = 0.1
    iou_th: float = 0.5
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
