from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    """
    Args:
        pos_weight: a weight for positive sample
        gamma: a hyperparameter for focusing the easy examples 
            (In gfocal loss paper, this is same role as the alpha)
        alpha: a hyperparameter to weight easy and hard samples
            (In gfocal loss paper, it does not exist)
        reduction: the output type that can be selected from none, sum and mean
        use_sigmoid: If the activation function of output layer is sigmoid, set it to False, 
            otherwise set it to True
    Forward:
        logits: a prediction with torch tensor type and has shape of (N, ...)
        labels: a label with torch tensor type and has shape of (M, ...)
    Examples:
        >>> loss_func = QualityFocalLoss()
        >>> outputs = model(images)
        >>> loss = loss_func(outputs, labels)
        >>> loss.backward()
    """
    def __init__(self, pos_weight: torch.Tensor, gamma=1.5, alpha=0.25, reduction='none', use_sigmoid=True):
        super(QFocalLoss, self).__init__()
        assert reduction in ('mean', 'sum', 'none')
        self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight) \
            if use_sigmoid else nn.BCELoss(pos_weight=pos_weight)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, labels):
        loss = self.loss_func(pred, true)

        logits_prob = torch.sigmoid(logits)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(labels - logits_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss