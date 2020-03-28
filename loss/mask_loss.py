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

    def __call__(self, seg_mask, seg_gtmask, seg_label):
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
        if not isinstance(seg_mask, torch.Tensor) or not isinstance(seg_gtmask, torch.Tensor):
            return 0

        # 截断出末尾作为输入
        if seg_mask.shape[0] > seg_label.shape[0]:
            seg_mask = seg_mask[seg_mask.shape[0] - seg_label.shape[0]:seg_mask.shape[0]]

        if seg_mask.shape[1] > seg_gtmask.shape[1]:
            seg_mask = seg_mask[:, seg_mask.shape[1] - seg_gtmask.shape[1]:seg_mask.shape[1]]

        loss = F.binary_cross_entropy(seg_mask, seg_gtmask, reduction="none")

        # 病灶并集处要分为一组
        seg_mask_max = torch.max(seg_gtmask, dim=1, keepdim=True)[0]
        seg_gtmask = seg_mask_max.expand_as(seg_gtmask)

        pos_num = torch.sum(seg_gtmask)
        pos_loss_map = loss * seg_gtmask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map) / pos_num
        else:
            pos_loss = 0

        neg_num = torch.sum((1 - seg_gtmask))
        neg_loss_map = loss * (1 - seg_gtmask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map) / neg_num
        else:
            neg_loss = 0
        total_loss = pos_loss + neg_loss

        return total_loss

# 原本是要让分割网络单独输出一层学习网络的CAM，但是现在想想还是放弃了，我新开一个直接监督网络CAM吧
class GradCamMaskLoss01(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    def __init__(self, seg_num_classes):
        self.seg_num_classes = seg_num_classes
        pass

    def __call__(self, output_mask, gcam_mask, label, seg_mask):   #output_mask, seg_mask, seg_label
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

        if output_mask.shape[1] > self.seg_num_classes:
            output_mask = output_mask[:, 0:output_mask.shape[1]-self.seg_num_classes]

        if output_mask.shape[1] != gcam_mask.shape[1]:  #gcam_mask.shape[1] == 1
            l = []
            for i in range(label.shape[0]):
                l.append(output_mask[i][label[i]].unsqueeze(0).unsqueeze(0))
            output_mask = torch.cat(l, dim=0)
            l.clear()

        output_score = torch.sigmoid(output_mask)

        #loss = F.binary_cross_entropy(output_score, gcam_mask, reduction="none")
        #loss = torch.abs(output_score - gcam_mask)
        # KL
        loss = gcam_mask * torch.log(gcam_mask/output_score) + (1-gcam_mask) * torch.log((1-gcam_mask)/(1-output_score))
        # cross_entropy_loss 连续
        #loss = -(gcam_mask * torch.log(output_score) + (1 - gcam_mask) * torch.log(1 - output_score))

        """
        seg_mask_max = torch.max(gcam_mask, dim=1, keepdim=True)[0]
        seg_mask = seg_mask_max.expand_as(gcam_mask)

        # seed-loss
        pos_num = torch.sum(seg_mask)
        pos_loss_map = loss * seg_mask
        if pos_num != 0:
            pos_loss = torch.sum(pos_loss_map) / pos_num
        else:
            pos_loss = 0

        # 只取正类损失，实际即为seed-loss。那么剩余部分用neg-masked-img-loss监督以使范围不要太大
        
        neg_num = torch.sum((1 - seg_mask))
        neg_loss_map = loss * (1 - seg_mask)
        if neg_num != 0:
            neg_loss = torch.sum(neg_loss_map) / neg_num
        else:
            neg_loss = 0
        #"""
        #total_loss = pos_loss + neg_loss
        total_loss = torch.mean(loss)
        return total_loss


class GradCamMaskLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, seg_num_classes):
        self.seg_num_classes = seg_num_classes
        pass

    def __call__(self, gcam_mask_list, gcam_gtmask, gcam_label):  # output_mask, seg_mask, seg_label
        #  CJY distribution 1
        # 用真值sef_mask监督CAM
        # """
        if not isinstance(gcam_gtmask, torch.Tensor) or not isinstance(gcam_mask_list, list):
            return [0, 0, 0, 0]

        # seg_mask 需要根据病灶重新生成分级所需要的掩膜
        #"""
        NewSegMask = []
        for i in range(gcam_label.shape[0]):
            if gcam_label[i] == 1:
                sm_p = gcam_gtmask[i:i + 1, 2:3]

                sm_n1 = gcam_gtmask[i:i + 1, 0:2]
                sm_n2 = gcam_gtmask[i:i + 1, 3:4]
                sm_n = torch.cat([sm_n1, sm_n2], dim=1)
                sm_n = 1 - sm_p

                sm_un = sm_p #1-torch.max(gcam_gtmask)[0]

            elif gcam_label[i] == 2:
                sm_p1 = gcam_gtmask[i:i + 1, 0:2]
                sm_p2 = gcam_gtmask[i:i + 1, 3:4]
                sm_p = torch.cat([sm_p1, sm_p2], dim=1)
                #sm = seg_mask[i:i + 1, 0:4]  #还是应该去除2  #但是有一个样本有问题，他只有MA，但是分为了grade2

                sm_n = gcam_gtmask[i:i + 1, 2:3]
                sm_n = 1 - sm_p

                sm_un = sm_p  # 1-torch.max(gcam_gtmask)[0]

            elif gcam_label[i] == 3:
                sm_p = gcam_gtmask[i:i + 1, 1:2]

                sm_n1 = gcam_gtmask[i:i + 1, 0:1]
                sm_n2 = gcam_gtmask[i:i + 1, 2:4]
                sm_n = torch.cat([sm_n1, sm_n2], dim=1)  # 对于3，4不能如此，因为其他位置可能会有别的病灶，不能掩盖，最后一层定位不准确

                sm_un = 1-torch.max(gcam_gtmask, dim=1, keepdim=True)[0]

            elif gcam_label[i] == 4:
                sm_p = gcam_gtmask[i:i + 1, 1:2]

                sm_n1 = gcam_gtmask[i:i + 1, 0:1]
                sm_n2 = gcam_gtmask[i:i + 1, 2:4]
                sm_n = torch.cat([sm_n1, sm_n2], dim=1)  # 对于3，4不能如此，因为其他位置可能会有别的病灶，不能掩盖，最后一层定位不准确

                sm_un = 1 - torch.max(gcam_gtmask, dim=1, keepdim=True)[0]
            else:  # 如果不是1-4级，就不要用于监督了，放弃该样本
                continue

            sm_p = torch.max(sm_p, dim=1, keepdim=True)[0]
            sm_n = torch.max(sm_n, dim=1, keepdim=True)[0]

            sm = torch.cat([sm_p*3, sm_un*2, sm_n], dim=1)
            sm = torch.max(sm, dim=1, keepdim=True)[0]  # 那么sm就是-1：抑制  1：激活  0：未知
            sm = (sm - 1)/2
            NewSegMask.append(sm)
        gcam_gtmask = torch.cat(NewSegMask, dim=0)
        #"""
        # CJY at 2020.3.26
        # 或许，我不该挑出单独的病灶，因为对于高级别的病，也会有低级别的病灶，这样可能会产生混淆
        #gcam_gtmask = torch.max(gcam_gtmask, dim=1, keepdim=True)[0]

        total_loss_list = []
        seg_mask_c = gcam_gtmask
        # 遍历所有生成的gcam_mask
        for gcam_mask in reversed(gcam_mask_list):
            if gcam_mask.shape[0] >= seg_mask_c.shape[0]:
                gcam_mask = gcam_mask[gcam_mask.shape[0] - seg_mask_c.shape[0]:gcam_mask.shape[0]]
            else:
                raise Exception("output_mask.shape[0] can't match label.shape[0]")

            # 需要将seg_mask根据gcam_mask的大小进行调整
            if gcam_mask.shape[-1] != seg_mask_c.shape[-1]:
                gcam_gtmask = F.adaptive_max_pool2d(seg_mask_c, (gcam_mask.shape[-2], gcam_mask.shape[-1]))

            #gcam_mask_p = gcam_mask * gcam_gtmask
            #gcam_mask_n = torch.relu(gcam_mask * (1 - gcam_gtmask))
            #gcam_mask = gcam_mask_p + gcam_mask_n

            # 计算交叉熵损失
            #loss = F.binary_cross_entropy(gcam_mask, gcam_gtmask, reduction="none")
            loss = torch.pow(gcam_mask - gcam_gtmask, 2)


            # 只取seg_mask为1的位置处的loss计算 因为为0的位置处不清楚重要性
            region1 = torch.ne(gcam_gtmask, 0.5).float() #* torch.gt(gcam_mask, 1)
            pos_num = torch.sum(region1)
            pos_loss_map = loss * region1
            if pos_num != 0:
                pos_loss = torch.sum(pos_loss_map) / pos_num
            else:
                pos_loss = 0

            # """
            region2 = torch.eq(gcam_gtmask, 0).float() #F.max_pool2d(seg_mask, kernel_size=11, stride=1, padding=5)
            neg_num = torch.sum(region2)
            neg_loss_map = loss * (region2)
            if neg_num != 0:
                neg_loss = torch.sum(neg_loss_map) / neg_num
            else:
                neg_loss = 0
            # """

            # a = torch.isnan(pos_loss)
            # if a.item() == 1:
            #    print("Nan")
            total_loss_list.append(pos_loss)

        while len(total_loss_list) < 4:
            total_loss_list.append(0)

        return total_loss_list