import torch.nn as nn
from V2.UTILS.dataset_define import *
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer


trainer_opt = TrainConfig()
data_opt = DataSetConfig()

dark_net = DarkNet19()
net = YOLOV2Net(dark_net)
net = nn.DataParallel(net)
net.cuda()

trainer = YOLOV2Trainer(
    net,
    data_opt,
    trainer_opt
)
image_net_224_train_loader, image_net_224_test_loader = get_image_net_224_loader(data_opt, trainer_opt)

trainer.train_on_image_net_224(
    image_net_224_train_loader, image_net_224_test_loader
)

image_net_448_train_loader, image_net_448_test_loader = get_image_net_448_loader(data_opt, trainer_opt)
trainer.train_on_image_net_448(
    image_net_448_train_loader, image_net_448_test_loader
)

voc_train_loader, voc_test_loader = get_voc_data_loader(data_opt, trainer_opt)
trainer.train_object_detector(
    voc_train_loader, voc_test_loader
)
