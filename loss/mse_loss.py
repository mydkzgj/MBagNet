#CJY 处理多标签分类  come from CrossEntropyLabelSmooth

import torch
from torch import nn

class MSELoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction="none")

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        targets = targets.float()
        loss = self.MSE(inputs, targets)
        #loss = torch.mean(loss)

        # CJY at 2020.9.17 为了配合回归  版本二
        input_target_both_le_zero = ~ (inputs.le(0) * targets.le(0))
        loss = torch.mean(loss * input_target_both_le_zero.float())
        return loss
