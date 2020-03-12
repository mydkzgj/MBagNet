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


class SegMaskLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self, seg_num_classes):
        self.seg_num_classes = seg_num_classes
        pass

    def __call__(self, output_mask, seg_mask, seg_label):   #output_mask, seg_mask, seg_label
        #  CJY distribution 1
        """
        if not isinstance(output_mask, torch.Tensor):
            return 0

        if output_mask.shape[1] >= 4:
            output_mask = output_mask[:, 0:4]

        output_score = torch.sigmoid(output_mask)
        loss = F.binary_cross_entropy(output_score, seg_mask, reduction="none")

        pos_num = torch.sum(seg_mask)
        pos_loss_map = loss * seg_mask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map)/pos_num
        else:
            pos_loss = 0

        neg_num = torch.sum((1 - seg_mask))
        neg_loss_map = loss * (1 - seg_mask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map)/neg_num
        else:
            neg_loss = 0

        total_loss = pos_loss + neg_loss
        #"""

        # CJY distribution 2
        #"""
        # 注意：负样本的数量实在太多，会淹没误判的正样本。
        # 该版本要比第一种好
        if not isinstance(output_mask, torch.Tensor) or not isinstance(seg_mask, torch.Tensor):
            return 0

        # 截断出末尾作为输入
        if output_mask.shape[0] > seg_label.shape[0]:
            output_mask = output_mask[output_mask.shape[0]-seg_label.shape[0]:output_mask.shape[0]]

        if output_mask.shape[1] > self.seg_num_classes:
            output_mask = output_mask[output_mask.shape[1]-self.seg_num_classes:output_mask.shape[1]]

        output_score = torch.sigmoid(output_mask)
        loss = F.binary_cross_entropy(output_score, seg_mask, reduction="none")

        # 病灶并集处要分为一组
        seg_mask_max = torch.max(seg_mask, dim=1, keepdim=True)[0]
        seg_mask = seg_mask_max.expand_as(seg_mask)

        pos_num = torch.sum(seg_mask)
        pos_loss_map = loss * seg_mask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map) / pos_num
        else:
            pos_loss = 0

        neg_num = torch.sum((1 - seg_mask))
        neg_loss_map = loss * (1 - seg_mask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map) / neg_num
        else:
            neg_loss = 0
        total_loss = pos_loss + neg_loss

        return total_loss

class GradCamMaskLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self):
        pass

    def __call__(self, output_mask, gcam_mask, label):   #output_mask, seg_mask, seg_label
        #  CJY distribution 1
        """
        if not isinstance(output_mask, torch.Tensor):
            return 0

        if output_mask.shape[1] >= 4:
            output_mask = output_mask[:, 0:4]

        output_score = torch.sigmoid(output_mask)
        loss = F.binary_cross_entropy(output_score, seg_mask, reduction="none")

        pos_num = torch.sum(seg_mask)
        pos_loss_map = loss * seg_mask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map)/pos_num
        else:
            pos_loss = 0

        neg_num = torch.sum((1 - seg_mask))
        neg_loss_map = loss * (1 - seg_mask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map)/neg_num
        else:
            neg_loss = 0

        total_loss = pos_loss + neg_loss
        #"""

        # CJY distribution 2
        #"""
        # 注意：负样本的数量实在太多，会淹没误判的正样本。
        # 该版本要比第一种好
        if not isinstance(output_mask, torch.Tensor) or not isinstance(gcam_mask, torch.Tensor):
            return 0

        if output_mask.shape[0] <= label.shape[0]:
            label = label[label.shape[0]-output_mask.shape[0]:label.shape[0]]
        else:
            raise Exception("output_mask.shape[0] can't match label.shape[0]")

        if output_mask.shape[1] != gcam_mask.shape[1]:  #gcam_mask.shape[1] == 1
            l = []
            for i in range(label.shape[0]):
                l.append(output_mask[i][label[i]].unsqueeze(0).unsqueeze(0))
            output_mask = torch.cat(l, dim=0)
            l.clear()

        output_score = torch.sigmoid(output_mask)
        loss = F.binary_cross_entropy(output_score, gcam_mask, reduction="none")

        # 舍弃分组吧
        """
        seg_mask_max = torch.max(gcam_mask, dim=1, keepdim=True)[0]
        seg_mask = seg_mask_max.expand_as(gcam_mask)
        
        pos_num = torch.sum(seg_mask)
        pos_loss_map = loss * seg_mask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map) / pos_num
        else:
            pos_loss = 0

        neg_num = torch.sum((1 - seg_mask))
        neg_loss_map = loss * (1 - seg_mask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map) / neg_num
        else:
            neg_loss = 0
        total_loss = pos_loss + neg_loss
        """
        total_loss = torch.mean(loss)

        return total_loss

