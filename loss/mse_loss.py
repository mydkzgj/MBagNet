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
        self.MSE = torch.nn.MSELoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        targets = targets.float()
        inputs2 = torch.relu(inputs)
        targets2 = targets
        loss = self.MSE(inputs2, targets2)
        #loss = self.BCE(torch.relu(torch.tanh(inputs)), targets)
        return loss
