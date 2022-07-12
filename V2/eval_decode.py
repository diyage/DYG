
from UTILS import CV2, XMLTranslate
from V2.UTILS.position_translate import Position, PositionTranslate
from V2.UTILS.others import YOLOV2Tools
from V2.UTILS.config_define import *


x = XMLTranslate('E:/yolo/', '2007_000032.xml')
x.resize()
img = x.img

labels = []
now_label = []
for obj in x.get_objects():

    tmp = (obj[0], obj[1], obj[2], obj[3], obj[4])
    now_label.append(tmp)

labels.append(now_label)
res = YOLOV2Tools.make_targets(
    labels,
    DataSetConfig.pre_anchor_w_h,
    DataSetConfig.image_size,
    DataSetConfig.grid_number,
    DataSetConfig.kinds_name
)
# pre = YOLOV2Tools.decode_out(
#     res[0],
#     DataSetConfig.pre_anchor_w_h,
#     DataSetConfig.image_size,
#     DataSetConfig.grid_number,
#     DataSetConfig.kinds_name,
#     use_conf=False,
#     use_score=False,
#     out_is_target=True
# )
# print(labels)
# print(pre)
position, conf, scores = YOLOV2Tools.split_output(
    res,
    len(DataSetConfig.pre_anchor_w_h)
)
abs_pos = YOLOV2Tools.translate_to_abs_position(
    position,
    DataSetConfig.pre_anchor_w_h,
    DataSetConfig.image_size,
    DataSetConfig.grid_number
)
