import torch.nn.functional as F

def soft_label_cross_entropy(input, target, weights):
    return  -(target * input).sum() / input.shape[0]

def hard_label_cross_entropy(output, target, weights):
    return F.nll_loss(output, target.long().squeeze(), weight=weights)
