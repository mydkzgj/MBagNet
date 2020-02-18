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


class MaskLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self):
        pass

    def __call__(self, seg_mask, label_mask, seg_label):
        #seg_mask = torch.abs(seg_mask)  #abs

        """
        N = seg_label.shape[0]
        csmlist = []
        for i in range(N):
            #class_seg_mask_score = seg_mask[i, 0:5]   #取前几维
            templist = []
            for j in range(5):
                if j == seg_label[i].item():
                    class_seg_mask_score = seg_mask[i, j].unsqueeze(0).unsqueeze(0)
                else:
                    class_seg_mask_score = -seg_mask[i, j].unsqueeze(0).unsqueeze(0)
                templist.append(class_seg_mask_score)

            #class_seg_mask_score = seg_mask[i,seg_label[i]].unsqueeze(0).unsqueeze(0)
            #class_seg_mask_normal = -seg_mask[i, 0].unsqueeze(0).unsqueeze(0)
            class_seg_mask_score = torch.cat(templist, dim=1)
            csmlist.append(class_seg_mask_score)

        csm = torch.cat(csmlist, dim=0)
        csm_score = torch.tanh(csm)

        label_mask = label_mask.expand(label_mask.shape[0], label_mask.shape[1]*5, label_mask.shape[2], label_mask.shape[3])
        """
        """
        # 依据label_mask对像素进行分组
        groudtruth_pos_map = label_mask.bool()
        csm_gpos = csm[groudtruth_pos_map]
        csm_gneg = csm[groudtruth_pos_map]

        #应该正的值中小于1的值
        csm_gpos_neg1_index = torch.lt(csm_gpos, 0)
        csm_gpos_neg1 = csm_gpos[csm_gpos_neg1_index]
        loss_pos = 1 - csm_gpos_neg1

        #应该负的值中大于0的值
        csm_gneg_pos0_index = torch.gt(csm_gneg, 0)
        csm_gneg_pos0 = csm_gpos[csm_gneg_pos0_index]
        loss_neg = csm_gneg_pos0
        """

        #loss = F.binary_cross_entropy(csm_score, label_mask, reduction="none")

        #csm_score = torch.sigmoid(seg_mask) #torch.tanh(torch.relu(seg_mask))
        #loss = F.binary_cross_entropy(csm_score, label_mask, reduction="none")

        #loss_with_pos_weight = loss * (label_mask * 9 + 1)

        """

        #predict_pos_map = torch.gt(csm, 0)
        groudtruth_pos_map = label_mask.bool()
        #neg_mask = (~groudtruth_pos_map) & predict_pos_map

        loss_pos = loss[groudtruth_pos_map]
        #loss_neg = loss[neg_mask]
        #loss_neg = loss[~groudtruth_pos_map]

        loss_neg = csm_score[~groudtruth_pos_map]

        if loss_pos.shape[0] != 0:
            loss_pos_mean = torch.mean(loss_pos)
        else:
            #print("loss_pos:", loss_pos)
            loss_pos_mean = 0

        if loss_neg.shape[0] != 0:
            loss_neg_mean = torch.mean(loss_neg)
        else:
            #print("loss_neg:", loss_neg)
            loss_neg_mean = 0


        total_loss = loss_pos_mean + loss_neg_mean
        """
        #total_loss = torch.mean(loss_with_pos_weight)
        #"""
        #loss = torch.pow(seg_mask[:, 0:4] - label_mask, 2)
        loss = F.binary_cross_entropy(torch.sigmoid(seg_mask), label_mask, reduction="none")

        #loss_pos = torch.sum(loss * label_mask) / torch.sum(label_mask)
        #loss_neg = torch.sum(loss * (1-label_mask)) / torch.sum(1-label_mask)
        #total_loss = loss_pos + loss_neg
        #total_loss = torch.mean(loss)

        #loss_with_pos_weight = loss * (label_mask * 99 + 1)
        #total_loss = torch.mean(loss_with_pos_weight)

        loss_weight = label_mask/(torch.sum(label_mask)) + (1-label_mask)/(torch.sum(1-label_mask))
        loss = loss * loss_weight
        total_loss = torch.sum(loss)
        #"""
        #loss_weight = label_mask / (torch.sum(label_mask)) + (1 - label_mask) / (torch.sum(1 - label_mask))
        #loss = torch.pow(seg_mask - label_mask, 2)
        #total_loss = torch.sum(loss)

        return total_loss

