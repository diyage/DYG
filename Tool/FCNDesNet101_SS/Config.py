import numpy as np


class SSDataSetConfig:
    root_path: str = '/home/dell/data/DataSet/VOC/'
    years: list = ['2007', '2012']
    # data set root dir
    image_size: tuple = (608, 608)

    mean = [0.406, 0.456, 0.485]

    std = [0.225, 0.224, 0.229]


class SSTrainConfig:
    max_epoch_on_detector = 200
    num_workers: int = 4
    device: str = 'cuda:0'
    batch_size = 4
    lr: float = 1e-3
    warm_up_end_epoch: int = 5


class SSEvalConfig:
    eval_frequency = 10


class SSConfig:
    ABS_PATH: str = '/home/dell/data2/models/'
    # saved model or images dir(part of )
    # if you want save something to now path, please set ABS_PATH = ''
    data_config = SSDataSetConfig()
    train_config = SSTrainConfig()
    eval_config = SSEvalConfig()


