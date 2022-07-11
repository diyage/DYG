import torch
import torch.nn as nn
from V2.UTILS.others import YOLOV2Tools


class YOLOV2Loss(nn.Module):
    def __init__(self,
                 anchor_number: int = 5):
        super().__init__()
        self.anchor_number = anchor_number

    def split(self,
              x: torch.Tensor):

        N, C, H, W = x.shape
        K = C // self.anchor_number   # K = (x, y, w, h, conf, kinds0, kinds1, ...)
        # C = anchor_number * K
        x = x.view(N, self.anchor_number, K, H, W)
        x = x.permute(0, 3, 4, 1, 2)  # N * H * W * a_n * K

        position = x[:, :, :, :, 0:4]  # N * H * W * a_n * 4
        conf = x[:, :, :, :, 4].unsqueeze(-1)  # N * H * W * a_n * 1
        scores = x[:, :, :, :, 5:]  # N * H * W * a_n * ...

        return position, conf, scores

    def forward(self,
                out: torch.Tensor,
                gt: torch.Tensor):

        o_position, o_conf, o_scores = self.split(out)
        g_position, g_conf, g_scores = self.split(gt)
        N, H, W, a_n, _ = o_position.shape

        no_obj_response_mask = (g_conf == 0).float()  # N * H * W * a_n * 1
        # just used for conf(no obj)

        has_obj_response_mask = (g_conf > 0).float()  # N * H * W * a_n * 1
        # used for all (must  g_conf > 0 and  anchor box response IOU max )
        for i in range(N):
            for r in range(H):
                for c in range(W):
                    now_pred_position = o_position[i, r, c]  # an * 4
                    now_gt_position = g_position[i, r, c, 0].unsqueeze(0)  # 1 * 4


a = torch.rand(size=( 3, 4 )) - 0.5
print(a[0])