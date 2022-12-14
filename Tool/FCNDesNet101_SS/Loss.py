import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(
            self,
            gama: float = 2.0,
            compute_dim: int = 1
    ):
        super().__init__()
        self.gama = gama
        self.compute_dim = compute_dim

    def forward(
            self,
            pred: torch.Tensor,
            gt: torch.Tensor,
    ):

        pred_softmax = torch.softmax(pred, dim=self.compute_dim)

        weight = (1 - pred_softmax) ** self.gama

        loss = -1.0 * weight * gt * torch.log_softmax(pred, dim=self.compute_dim)
        """
        important, do not use weight.detach()
        """
        # loss = -1.0 * weight * gt * torch.log(pred_softmax)

        return {'total_loss': loss.mean()}
