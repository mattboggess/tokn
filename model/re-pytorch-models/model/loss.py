import torch.nn.functional as F


def nll_loss(output, target, weights):
    return F.nll_loss(output, target, weight=weights)
