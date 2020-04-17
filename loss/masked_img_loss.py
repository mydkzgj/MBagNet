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

    def __call__(self, pos_masked_logits, neg_masked_logits, origin_logits, label, ):   #output_mask, seg_mask, seg_label
        if not isinstance(pos_masked_logits, torch.Tensor):
            return 0
        """
        # CJY distribution 1  cross_entropy_loss min
        # pos_masked区域的img应该更容易区分类别
        reload_label = label[label.shape[0] - pos_masked_logits.shape[0]:label.shape[0]]
        loss = F.cross_entropy(pos_masked_logits, reload_label, reduction="none")

        #reload_label = label[label.shape[0] - neg_masked_logits.shape[0]:label.shape[0]]
        #one_hot_label = torch.nn.functional.one_hot(reload_label, pos_masked_logits.shape[1]).float()
        #score = -torch.log(F.softmax(pos_masked_logits, dim=1))
        #score = 1-F.softmax(pos_masked_logits, dim=1)
        #loss = score[one_hot_label.bool()]

        # 挑选指定sample的loss
        pick_index = torch.ne(reload_label, -1) & torch.ne(reload_label, 5)  & torch.ne(reload_label, 3) & torch.ne(reload_label, 4)#& torch.ne(label, 0)
        if pick_index.sum() == 0:
            return 0
        pick_loss = loss[pick_index]
        total_loss = torch.mean(pick_loss)
        #"""

        #"""
        # CJY distribution 2  logits diff min
        # 由pos_masked区域主要提供logit
        reload_label = label[label.shape[0]-pos_masked_logits.shape[0]:label.shape[0]]
        origin_logits = origin_logits[origin_logits.shape[0]-pos_masked_logits.shape[0]:origin_logits.shape[0]]

        # 1.只关注label对应的logits
        """
        one_hot_label = torch.nn.functional.one_hot(reload_label, pos_masked_logits.shape[1]).float()
        ori_logits = origin_logits[one_hot_label.bool()]
        pm_logits = pos_masked_logits[one_hot_label.bool()]
        #loss = torch.pow(pm_logits - ori_logits, 2)  # 只限制pm-logits好像不太好

        op_logits = torch.cat([ori_logits.unsqueeze(1), pm_logits.unsqueeze(1)], dim=1)
        max_opL = torch.max(op_logits.abs(), dim=1)[0].detach()

        loss = torch.pow((pm_logits - ori_logits) / max_opL, 2) * 0.5 #* max_opL
        #loss = torch.abs(pm_logits - ori_logits)/(torch.abs(ori_logits).clamp(min=1E-12).detach())    #相对距离
        #"""

        # 2.关注不同label的logits
        #"""
        one_hot_label = torch.nn.functional.one_hot(reload_label, pos_masked_logits.shape[1]).float()
        d_logits = pos_masked_logits - origin_logits.detach()
        d_logits = d_logits * (one_hot_label - 0.5) * (-2)
        d_logits = torch.relu(d_logits)
        loss = torch.sum(d_logits, dim=1)
        #"""

        # 3.输入为gcam-logits
        """
        #loss = torch.pow(pos_masked_logits - origin_logits, 2)
        #origin_logits = torch.relu(origin_logits)
        #pos_masked_logits = torch.relu(pos_masked_logits)
        op_logits = torch.cat([origin_logits.unsqueeze(1), pos_masked_logits.unsqueeze(1)], dim=1)
        max_opL = torch.max(op_logits.abs(), dim=1)[0].detach()

        loss = torch.pow((pos_masked_logits - origin_logits)/max_opL, 2) * 0.5 * max_opL

        loss = torch.mean(loss.view(loss.shape[0], -1), dim=-1)    
        #"""
        
        # 挑选指定sample的loss
        pick_index = torch.ne(reload_label, -1) & torch.ne(reload_label, 5) & torch.ne(reload_label, 3) & torch.ne(reload_label, 4)#& torch.ne(label, 0)
        if pick_index.sum() == 0:
            return 0
        pick_loss = loss[pick_index]
        total_loss = torch.mean(pick_loss)
        #"""

        return total_loss

class NegMaskedImgLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self):
        self.CEL = torch.nn.CrossEntropyLoss()
        pass

    def __call__(self, pos_masked_logits, neg_masked_logits, origin_logits, label):   #output_mask, seg_mask, seg_label
        if not isinstance(neg_masked_logits, torch.Tensor):
            return 0
        """
        # CJY distribution 1  cross_entropy_loss min
        # 
        reload_label = label[label.shape[0]-neg_masked_logits.shape[0]:label.shape[0]]
        loss = F.cross_entropy(neg_masked_logits, reload_label, reduction="none")

        # 挑选指定sample的loss
        pick_index = torch.ne(reload_label, -1) & torch.ne(reload_label, 5)  # & torch.ne(label, 0)
        pick_loss = loss[pick_index]
        total_loss = torch.mean(pick_loss)
        # """

        """
        # CJY distribution 2  logits diff min

        # 由pos_masked区域主要提供logit
        origin_logits = origin_logits[origin_logits.shape[0]-neg_masked_logits.shape[0]:origin_logits.shape[0]]
        reload_label = label[label.shape[0]-neg_masked_logits.shape[0]:label.shape[0]]
        one_hot_label = torch.nn.functional.one_hot(reload_label, neg_masked_logits.shape[1]).float()
        ori_logits = origin_logits[one_hot_label.bool()]
        nm_logits = neg_masked_logits[one_hot_label.bool()]
        loss = torch.abs(nm_logits)/torch.abs(ori_logits).clamp(min=1E-12)    #相对距离

        # 挑选指定sample的loss
        pick_index = torch.ne(reload_label, -1) & torch.ne(reload_label, 5) & torch.ne(reload_label, 3) & torch.ne(reload_label, 4)#& torch.ne(label, 0)
        if pick_index.sum() == 0:
            return 0
        pick_loss = loss[pick_index]
        total_loss = torch.mean(pick_loss)
        #"""

        """
        # CJY distribution 3  score min
        # 由pos_masked区域主要提供logit
        origin_logits = origin_logits[origin_logits.shape[0]-neg_masked_logits.shape[0]:origin_logits.shape[0]]
        reload_label = label[label.shape[0]-neg_masked_logits.shape[0]:label.shape[0]]
        one_hot_label = torch.nn.functional.one_hot(reload_label, neg_masked_logits.shape[1]).float()

        #score = -torch.log(1-F.softmax(neg_masked_logits, dim=1)) #torch.sigmoid(neg_masked_logits)#
        score = F.softmax(neg_masked_logits, dim=1)
        loss = score[one_hot_label.bool()]

        # 对于label1和label2，去除所有病灶后，应该让label之前的label的score之和最大
        #score_list = []
        #for i in range(score.shape[0]):
        #    s = score[i:i+1, 0:reload_label[i]]
        #    score_list.append(s.sum(dim=1))
        #score = torch.cat(score_list, dim=0)
        #loss = -torch.log(score)

        # 若grade1，2标出所有病灶，那么去除这些病灶后，label应该为0
        #reload_label0 = reload_label * 0
        #loss = F.cross_entropy(neg_masked_logits, reload_label0, reduction="none")


        #"""

        reload_label = label[label.shape[0]-pos_masked_logits.shape[0]:label.shape[0]]
        origin_logits = origin_logits[origin_logits.shape[0]-pos_masked_logits.shape[0]:origin_logits.shape[0]]
        # 1.只关注label对应的logits
        # """
        one_hot_label = torch.nn.functional.one_hot(reload_label, neg_masked_logits.shape[1]).float()
        ori_logits = origin_logits[one_hot_label.bool()]
        nm_logits = neg_masked_logits[one_hot_label.bool()]
        # loss = torch.pow(pm_logits - ori_logits, 2)  # 只限制pm-logits好像不太好

        on_logits = torch.cat([ori_logits.unsqueeze(1), nm_logits.unsqueeze(1)], dim=1)
        max_onL = torch.max(on_logits.abs(), dim=1)[0].detach()

        loss = torch.pow(nm_logits/ max_onL, 2) * 0.5  # * max_opL
        # loss = torch.abs(pm_logits - ori_logits)/(torch.abs(ori_logits).clamp(min=1E-12).detach())    #相对距离
        # """

        # 挑选指定sample的loss
        pick_index = torch.ne(reload_label, -1) & torch.ne(reload_label, 5) & torch.ne(reload_label, 3) & torch.ne(reload_label, 4)#& torch.ne(label, 0)
        if pick_index.sum() == 0:
            return 0
        pick_loss = loss[pick_index]
        total_loss = torch.mean(pick_loss)

        return total_loss





