import torch.nn as nn
from V2.UTILS.dataset_define import *
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer
from tqdm import tqdm


trainer_opt = TrainConfig()
data_opt = DataSetConfig()

dark_net = DarkNet19()
net = YOLOV2Net(dark_net)

trainer = YOLOV2Trainer(
    net,
    data_opt,
    trainer_opt
)
image_net_224_train_loader, image_net_224_test_loader = get_image_net_224_loader(data_opt, trainer_opt)

for batch_id, (images, labels) in tqdm(enumerate(image_net_224_test_loader), desc='eval classifier'):
    print(batch_id)
