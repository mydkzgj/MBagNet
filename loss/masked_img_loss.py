# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F


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
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def rank_loss(dist_mat, labels, margin, alpha, tval):
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
        is_pos = labels.eq(labels)
        is_pos[ind] = 0  # 对角线置0
        is_neg = labels.ne(labels)

        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, margin - alpha), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        """
        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval*(-1*an_less_alpha+alpha))
        an_weight_sum = torch.sum(an_weight) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_weight_sum)
        """

        total_loss = total_loss + loss_ap
    total_loss = total_loss * 1.0 / N
    return total_loss


"""
# loss = -熵
class CommonLoss(object):
    "Common_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, num_classes, margin=None, alpha=None, tval=None):
        self.margin = margin
        #self.alpha = alpha
        #self.tval = tval
        cuda0 = torch.device('cuda:0')
        self.vector_label = torch.tensor([1/num_classes], dtype=torch.float64, device=cuda0)
        print(self.vector_label)

    def __call__(self, scores, labels):
        #if normalize_feature:
        #    global_feat = normalize_rank(global_feat, axis=-1)

        score_softmax = F.softmax(scores, dim=1)
        score_logsoftmax = F.log_softmax(scores, dim=1)
        score_preH = score_logsoftmax.mul(score_softmax)
        common_loss = torch.mean(score_preH)  #  dim1在上求得每个样本的熵，在dim0上求得batch的   x(-1)   均匀分布时熵最大

        return common_loss
"""


class PosMaskedImgLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self):
        self.CEL = torch.nn.CrossEntropyLoss()
        pass

    def __call__(self, pos_masked_logits, neg_masked_logits, origin_logits, label, one_hot_label):   #output_mask, seg_mask, seg_label
        if not isinstance(pos_masked_logits, torch.Tensor):
            return 0
        # 因为mask主要给的1，2，3，4类的信息，所以只考虑1，2，3，4类
        ori_logits = origin_logits[:, 1:5]

        #pos_mask
        pm_logits = pos_masked_logits[:, 1:5]
        pd_logits = torch.abs(pm_logits - ori_logits)/torch.abs(ori_logits).clamp(min=1E-12)    #相对距离
        pos_loss = torch.mean(pd_logits)

        total_loss = pos_loss

        return total_loss

class NegMaskedImgLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self):
        self.CEL = torch.nn.CrossEntropyLoss()
        pass

    def __call__(self, pos_masked_logits, neg_masked_logits, origin_logits, label, one_hot_label):   #output_mask, seg_mask, seg_label
        if not isinstance(neg_masked_logits, torch.Tensor):
            return 0
        # 因为mask主要给的1，2，3，4类的信息，所以只考虑1，2，3，4类
        
        ori_logits = origin_logits[:, 1:5]

        nm_logits = neg_masked_logits[:, 1:5]
        nd_logits = torch.abs(nm_logits)/torch.abs(ori_logits).clamp(min=1E-12)
        neg_loss = torch.mean(nd_logits)

        total_loss = neg_loss


        return total_loss





