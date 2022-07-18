import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

num_classes = 20
class_colors = [
    (np.random.randint(255), np.random.randint(255), np.random.randint(255)) for _ in range(num_classes)
]
print(class_colors)