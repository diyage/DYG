
from UTILS import CV2, XMLTranslate
from V2.UTILS.position_translate import Position, PositionTranslate
from V2.UTILS.others import YOLOV2Tools
from V2.UTILS.config_define import *

#
# x = XMLTranslate('/home/dell/下载/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/', '2007_000032.xml')
# x.resize(new_size=(416, 416))
# img = x.img
#
# labels = []
# now_label = []
# for obj in x.get_objects():
#
#     tmp = (obj[0], obj[1], obj[2], obj[3], obj[4])
#     now_label.append(tmp)
#
# labels.append(now_label)
# res = YOLOV2Tools.make_targets(
#     labels,
#     DataSetConfig.pre_anchor_w_h,
#     DataSetConfig.image_size,
#     DataSetConfig.grid_number,
#     DataSetConfig.kinds_name
# )
# pre = YOLOV2Tools.decode_out(
#     res[0],
#     DataSetConfig.pre_anchor_w_h,
#     DataSetConfig.image_size,
#     DataSetConfig.grid_number,
#     DataSetConfig.kinds_name,
#     use_conf=False,
#     use_score=True,
#     out_is_target=True
# )
#
# YOLOV2Tools.visualize(
#     x.img,
#     predict_name_pos_score=pre,
#     saved_path='pre.png'
# )

import torch.nn as nn
from V2.UTILS.dataset_define import *
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

trainer_opt = TrainConfig()
data_opt = DataSetConfig()

voc_train_loader, voc_test_loader = get_voc_data_loader(data_opt, trainer_opt)

trainer_opt = TrainConfig()
data_opt = DataSetConfig()

dark_net = DarkNet19()
net = YOLOV2Net(dark_net)

trainer = YOLOV2Trainer(
    net,
    data_opt,
    trainer_opt
)
trainer.show(voc_test_loader, 'tmp/')