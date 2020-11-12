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
    def __init__(self, foucusOnNegtiveLabel=False):
        super(MultilabelBinaryCrossEntropy, self).__init__()
        self.BCEL = torch.nn.BCEWithLogitsLoss(reduction="none")
        #self.BCE = torch.nn.BCELoss()
        self.foucusOnNegtiveLabel = foucusOnNegtiveLabel   #只计算0label的样本，大于0的样本交给MSE_LOSS

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """

        targets = targets.float()
        loss = self.BCEL(inputs, targets)

        if self.foucusOnNegtiveLabel == False:
            loss = torch.mean(loss)
        else:
            # CJY at 2020.9.17 为了配合回归  版本二
            input_target_both_gt_zero = ~ (inputs.ge(0) * targets.gt(0))
            loss = torch.mean(loss * input_target_both_gt_zero.float())

        return loss
