#CJY 处理多标签分类  come from CrossEntropyLabelSmooth

import torch
from torch import nn

class MultilabelBinaryCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self):
        super(MultilabelBinaryCrossEntropy, self).__init__()
        self.BCEL = torch.nn.BCEWithLogitsLoss()
        self.BCE = torch.nn.BCELoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        #log_probs = self.logsoftmax(inputs)

        #计算多标签对应的label
        #targets = targets/torch.sum(targets, dim=1, keepdim=True)

        #loss = (- targets * log_probs).mean(0).sum()

        targets = targets.float()
        #loss = self.BCEL(inputs, targets)
        loss = self.BCE(torch.relu(torch.tanh(inputs)), targets)
        return loss
