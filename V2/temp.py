import torch.nn as nn
from V2.UTILS.dataset_define import *
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer
import torch

torch.cuda.set_device(1)

trainer_opt = TrainConfig()
data_opt = DataSetConfig()

dark_net = DarkNet19()
net = YOLOV2Net(dark_net)

trainer = YOLOV2Trainer(
    net,
    data_opt,
    trainer_opt
)
trainer_opt.batch_size = 32
voc_train_loader, voc_test_loader = get_voc_data_loader(data_opt, trainer_opt)
trainer.train_object_detector(
    voc_train_loader, voc_test_loader
)
