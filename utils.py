import torch
import numpy as np
from datasets import Datasets


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def nll_loss(y_true, y_pred, eps = 1e-01):

    mu,var = y_pred
    return torch.nn.GaussianNLLLoss(eps=eps)(mu, y_true, var)

