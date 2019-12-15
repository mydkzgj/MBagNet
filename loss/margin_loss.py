# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn

class MarginLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
    def __init__(self, margin=None, alpha=None, tval=None):
        self.margin = 2
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, scores, one_hot_labels):

        one_hot_labels = one_hot_labels * 2 - 1   #将值变为-1,1
        new_scores = scores * one_hot_labels

        #pos_index = torch.ge(new_scores, self.margin)
        #neg_index = torch.lt(new_scores, self.margin)

        #neg_scores = new_scores[neg_index]
        #neg_scores = self.margin - neg_scores

        neg_scores = torch.clamp((new_scores-self.margin)*(-1), min=0.0)

        loss = torch.mean(neg_scores)


        return loss

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

