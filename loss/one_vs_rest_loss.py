# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F


class OneVsRestLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self):
        pass


    def __call__(self, scores, labels):   #要求label是one-hot类型
        one_hot_labels = torch.nn.functional.one_hot(labels, scores.shape[1]).float()

        loss = []
        num_classes = scores.shape[1]
        score = torch.split(scores, 1, dim=1)
        label = torch.split(one_hot_labels, 1, dim=1)

        #用交叉熵太严苛了，我只要求预测的那一类趋近于1，其余类
        for i in range(num_classes):
            loss.append(F.binary_cross_entropy(score[i], label[i]))
        return loss

