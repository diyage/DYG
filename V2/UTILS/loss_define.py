import torch
import torch.nn as nn
from V2.UTILS.others import YOLOV2Tools
import torch.nn.functional as F


class YOLOV2Loss(nn.Module):
    def __init__(self,
                 anchor_pre_wh: tuple,
                 weight_position: float = 1.0,
                 weight_conf_has_obj: float = 1.0,
                 weight_conf_no_obj: float = 1.0,
                 weight_score: float = 1.0):
        super().__init__()
        self.anchor_number = len(anchor_pre_wh)
        self.anchor_pre_wh = anchor_pre_wh
        self.weight_position = weight_position
        self.weight_conf_has_obj = weight_conf_has_obj
        self.weight_conf_no_obj = weight_conf_no_obj
        self.weight_score = weight_score
        self.mse = nn.MSELoss()

    def split(
            self,
            x: torch.Tensor
    ):
        return YOLOV2Tools.split_output(x, self.anchor_number)

    def xywh_xyxy(
            self,
            position: torch.Tensor
    ) -> torch.Tensor:
        # -1 * H * W * a_n * 4
        _, grid_number, _, _, _ = position.shape
        pre_wh = torch.tensor(self.anchor_pre_wh,
                              dtype=torch.float32).to(position.device)
        # a_n * 2

        a_b = position[..., 0:2]  # -1 * a_n * 2
        m_n = position[..., 2:4]  # -1 * a_n * 2

        center_x_y = torch.tanh(a_b)  # scaled on grid (and offset compare to now gird index !!! )
        w_h = torch.exp(m_n) * pre_wh.expand_as(m_n)  # scaled on grid

        x_y_0 = center_x_y - 0.5 * w_h
        x_y_0[x_y_0 < 0] = 0
        x_y_1 = center_x_y + 0.5 * w_h
        x_y_1[x_y_1 > grid_number] = grid_number
        return torch.cat((x_y_0, x_y_1), dim=-1)

    def forward(self,
                out: torch.Tensor,
                gt: torch.Tensor):
        # split output
        o_position, o_conf, o_scores = self.split(out)
        g_position, g_conf, g_scores = self.split(gt)
        N, H, W, a_n, _ = o_position.shape

        # translate position from xywh to xyxy(compare to grid)
        o_xyxy = self.xywh_xyxy(torch.clone(o_position.detach()))  # N * H * W * a_n * 4
        g_xyxy = self.xywh_xyxy(torch.clone(g_position.detach()))  # N * H * W * a_n * 4

        # compute iou
        iou = YOLOV2Tools.compute_iou(o_xyxy, g_xyxy)  # N * H * W * a_n
        iou_max_index = iou.max(-1)[1]  # N * H * W
        iou_max = F.one_hot(iou_max_index, a_n)   # N * H * W * a_n
        # there are a_n anchors but just one response(others not response) one-hot

        # compute response mask
        has_obj_response_mask = (g_conf > 0).float()  # N * H * W * a_n
        has_obj_response_mask = has_obj_response_mask * iou_max  # N * H * W * a_n
        # there are a_n anchors but just one response(others not response)
        has_obj_mask = has_obj_response_mask.bool()

        no_obj_response_mask = 1.0 - has_obj_response_mask  # N * H * W * a_n
        no_obj_mask = no_obj_response_mask.bool()

        mask = has_obj_mask.unsqueeze(-1).expand_as(o_position)
        position_loss = self.mse(o_position[mask], g_position[mask])

        mask = has_obj_mask
        has_obj_conf_loss = self.mse(o_conf[mask], g_conf[mask])

        mask = no_obj_mask
        no_obj_conf_loss = self.mse(o_conf[mask], g_conf[mask])

        mask = has_obj_mask.unsqueeze(-1).expand_as(o_scores)

        score_loss = self.mse(o_scores[mask], g_scores[mask])

        loss = self.weight_position * position_loss + \
            self.weight_conf_has_obj * has_obj_conf_loss + \
            self.weight_conf_no_obj * no_obj_conf_loss + \
            self.weight_score * score_loss

        return loss
