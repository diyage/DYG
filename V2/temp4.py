import torch
import torch.nn as nn
import torch.nn.functional as F

scores = torch.rand(size=(128, 20))
print(scores[0].max())
