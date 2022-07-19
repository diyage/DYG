import torch
from V2.UTILS.dataset_define import *
from V2.UTILS.get_pretrained_darknet_19 import DarkNet_19, get_pretained_dark_net_19
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.set_device(1)
trainer_opt = TrainConfig()
data_opt = DataSetConfig()

dark_net = get_pretained_dark_net_19('/home/dell/PycharmProjects/YOLO/Tool/darknet19_72.96.pth')
net = YOLOV2Net(dark_net)

trainer = YOLOV2Trainer(
    net,
    data_opt,
    trainer_opt
)


# image_net_224_train_loader, image_net_224_test_loader = get_image_net_224_loader(data_opt, trainer_opt)
# trainer.train_on_image_net_224(
#     image_net_224_train_loader, image_net_224_test_loader
# )
#
# image_net_448_train_loader, image_net_448_test_loader = get_image_net_448_loader(data_opt, trainer_opt)
# trainer.train_on_image_net_448(
#     image_net_448_train_loader, image_net_448_test_loader
# )


voc_train_loader, voc_test_loader = get_voc_data_loader(data_opt, trainer_opt)
# trainer.detector.load_state_dict(
#     torch.load('/home/dell/data2/models/home/dell/PycharmProjects/YOLO/V2/model_pth_detector/80.pth'))

trainer.train_object_detector(
    voc_train_loader, voc_test_loader
)
# trainer.show_detect_answer(
#     voc_train_loader,
#     saved_dir='tmp/'
# )

# if reconstruct this project：
# Five parts：YOLOV2Loss, Trainer, Predictor, Visualizer, Evaluator

#                    | YOLOV2Loss   ---> use model out_put and gt to compute loss
#                    | Trainer      ---> use train_dataloader to train the model
# YOLOV2BaseTools ---| Predictor    ---> decode the out_put to get kps(kind_name, position, scores)
#                    | Visualizer   ---> use kps to plot each figure
#                    | Evaluator    ---> compute mAP and accuracy
