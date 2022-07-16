import torch.nn as nn
from V2.UTILS.dataset_define import *
from V2.UTILS.model_define import DarkNet19, YOLOV2Net
from V2.UTILS.trainer import YOLOV2Trainer
from tqdm.auto import tqdm, trange
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#
# trainer_opt = TrainConfig()
# data_opt = DataSetConfig()
#
# dark_net = DarkNet19()
# net = YOLOV2Net(dark_net)
#
# trainer = YOLOV2Trainer(
#     net,
#     data_opt,
#     trainer_opt
# )
# image_net_224_train_loader, image_net_224_test_loader = get_image_net_224_loader(data_opt, trainer_opt)
import torch
grid_number = (13, 13)
N = 2
a_n = 5
index = torch.tensor(list(range(grid_number[0])), dtype=torch.float32)
grid_r, grid_c = torch.meshgrid(index, index)
grid = torch.cat((grid_r.unsqueeze(-1), grid_c.unsqueeze(-1)), dim=-1)
# H * W * 2

grid = grid.unsqueeze(-2).unsqueeze(0).expand(N,
                                              grid_number[0],
                                              grid_number[1],
                                              a_n,
                                              2)
t = grid_r.expand_as(torch.zeros(size=(3, 4, 13, 13)))
print(t[0, 0])
print(grid_r)
# for i in range(N):
#     for r in range(grid_number[0]):
#         for c in range(grid_number[1]):
#             for a in range(a_n):
#                 print('*'*10)
#                 print(r, c)
#                 print(grid[i, r, c, a])
