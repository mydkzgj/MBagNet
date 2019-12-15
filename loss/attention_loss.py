# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    xy = torch.mm(x, y.t())
    dist.addmm_(1, -2, x, y.t())
    #dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability   这样收敛慢？我去掉试试
    #CJY at 2019.10.15
    dist = dist.clamp(min=1e-12)
    return dist

def rank_loss(dist_mat, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        
        total_loss = total_loss+loss_ap+loss_an
    total_loss = total_loss*1.0/N
    return total_loss

def diff_distribute(global_feat, labels):
    N = global_feat.size(0)
    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])

        #feat_ap = global_feat[is_pos]
        #找到异类的特征向量
        feat_an = global_feat[is_neg]

        #找到该类的大于0.5的点的位置
        anthor_feat_pos_idx = global_feat[ind].gt(0.5).expand_as(feat_an)

        feat_an_p = feat_an[anthor_feat_pos_idx]

        loss_single = torch.sum(feat_an_p)/(feat_an_p.shape[0] + 1)

        #loss_single = torch.sum((feat_an + global_feat[ind] - 1).clamp(0))

        total_loss = total_loss + loss_single
    total_loss = total_loss * 1.0 / N
    return total_loss

class AttentionLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
    def __init__(self, margin=None, alpha=None, tval=None):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, normalize_feature=True):
        """
        #distribute loss

        #distribute_loss = 0.25 - torch.pow(global_feat - 0.5, 2)
        #distribute_loss = torch.sum(distribute_loss)/(global_feat.shape[0]*global_feat.shape[1])

        #feat_sum = global_feat.sum()/(global_feat.shape[0])#*global_feat.shape[1])

        pos_feat_idx = global_feat.gt(0.5)
        pos_feat = global_feat[pos_feat_idx]
        pos_loss = torch.sum(1 - pos_feat)

        neg_feat_idx = global_feat.le(0.5)
        neg_feat = global_feat[neg_feat_idx]
        neg_loss = torch.sum(neg_feat)

        distribute_loss = (pos_loss + neg_loss)/(global_feat.shape[0]*global_feat.shape[1])

        feat_sum = 10 * global_feat.sum()/(global_feat.shape[0]*global_feat.shape[1])

        #希望 异类的分布1的部分不发生重叠
        diff_loss = diff_distribute(global_feat, labels)
        """


        """
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)

        dist_mat = euclidean_dist_rank(global_feat, global_feat)
        total_loss = rank_loss(dist_mat,labels,self.margin,self.alpha,self.tval)
        #"""

        #CJY at 2019.11.12
        n, _, h, w = global_feat.shape
        #x = global_feat
        sum = torch.sum(global_feat.view(n, 1, h*w), dim=-1).clamp(min=1e-12).unsqueeze(-1).unsqueeze(-1).expand_as(global_feat)
        global_feat = global_feat/sum
        attention_norm = torch.norm(global_feat.view(n, 1, h*w), 2, dim=-1, keepdim=True)
        loss = 1 - torch.sum(attention_norm)/n

        return loss#distribute_loss+feat_sum+diff_loss

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

