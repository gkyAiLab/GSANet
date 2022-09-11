import torch
import torch.nn as nn
import math
import numpy as np

class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss

class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def mu_tonemap(hdr_image, mu=5000):
    return torch.log(1 + mu * hdr_image) / math.log(1 + mu)


class mu_loss(nn.Module):
    def __init__(self, gamma=2.24, percentile=99):
        self.gamma = gamma
        self.percentile = percentile

    def forward(self, pred, label):
        hdr_linear_ref = pred ** self.gamma
        hdr_linear_res = label ** self.gamma
        norm_perc = np.percentile(hdr_linear_ref.data.cpu().numpy().astype(np.float32), self.percentile)
        mu_pred = tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc)
        mu_label = tanh_norm_mu_tonemap(hdr_linear_res, norm_perc)
        return nn.L1Loss()(mu_pred, mu_label)