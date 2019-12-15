# encoding: utf-8
"""
@author:  zzg
@contact: xhx1247786632@gmail.com
"""
import torch
from torch import nn
import torch.nn.functional as F
#from .reanked_loss import rank_loss
import math
from torch.nn import Module


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

def cosine_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    feat_num = x.size(1)
    m, n = x.size(0), y.size(0)

    xx = x.expand(n, m, feat_num)
    yy = y.expand(m, n, feat_num).transpose(1,0)

    dist = F.cosine_similarity(xx, yy, dim=2)
    return dist


def rank_loss(dist_mat, labels, alpha, beta, tval):
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
        # 1.避免重复计算； 2，避免梯度抵消
        for i in range(ind+1):
            is_pos[i] = 0
            is_neg[i] = 0

        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        ap_is_pos = torch.clamp(torch.add(dist_ap, - beta), min=0.0)
        ap_pos_num = ap_is_pos.size(0) + 1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos)
        loss_ap = torch.div(ap_pos_val_sum, float(ap_pos_num))

        an_is_pos = torch.lt(dist_an, alpha)
        an_less_alpha = dist_an[an_is_pos]
        an_weight = torch.exp(tval * (-1 * an_less_alpha + alpha))
        an_weight_sum = torch.sum(an_weight) + 1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm, an_weight))
        loss_an = torch.div(an_ln_sum, an_weight_sum)

        total_loss = total_loss + loss_ap + loss_an
    total_loss = total_loss * 1.0 / N
    return total_loss

class ClusterLoss(Module):#(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"

    def __init__(self, num_classes=2, feat_dim=2048, r_outer=1, io_ratio=3, distance_type="euclidean", use_gpu=True):
        super(ClusterLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.io_ratio = io_ratio

        self.tval = 1.0

        self.distance_type = distance_type
        self.use_gpu = use_gpu

        self.center_lock = 0

        """
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(1, self.feat_dim).cuda())   #将Center作为一种参数
            #self.R_P = nn.Parameter(torch.tensor([10]).float().cuda())
            self.r_P = nn.Parameter(torch.tensor([0.1]).float().cuda())
        else:
            self.centers = nn.Parameter(torch.randn(1, self.feat_dim))
            #self.R_P = nn.Parameter(torch.tensor([10]).float())
            self.r_P = nn.Parameter(torch.tensor([0.1]).float())
        """
        if self.use_gpu:
            self.center_buffer = torch.zeros(1, self.feat_dim).cuda()
            self.r_outer = torch.tensor([r_outer]).float().cuda()
            self.r_inter = self.r_outer * self.io_ratio
            self.d_center = torch.randn(1, self.feat_dim).cuda()
            self.d_r_outer = torch.zeros(1).cuda()
        else:
            self.center_buffer = torch.randn(1, self.feat_dim)
            self.r_outer = torch.tensor([r_outer]).float()
            self.r_inter = self.r_outer * self.io_ratio
            self.d_center = torch.randn(1, self.feat_dim)
            self.d_r_outer = torch.zeros(1)

        if self.distance_type == "euclidean":
            self.alpha = 2 * ((self.r_outer * (self.r_outer + self.r_inter)).sqrt() + self.r_outer) #- self.r_outer) # 两类之间的最小距离  此处不减去r了，+r
            self.beta = 2 * self.r_outer  # 类内最大距离  直径  #与ranked_loss 定义的参数稍有不同  margin和beta可以互换  margin = alpha - beta
        elif self.distance_type == "cos":
            sin_a_half = self.r_outer/(self.r_outer+self.r_inter)
            cos_a_half = ((self.r_inter + 2* self.r_outer) * self.r_inter ).sqrt()/(self.r_inter + self.r_outer)    #>0
            cos_a = 1-2*sin_a_half.pow(2)
            sin_a = 2*sin_a_half*cos_a_half
            cos_b = (self.r_inter - self.r_outer)/(self.r_inter + self.r_outer)
            sin_b = (4 * self.r_inter*self.r_outer).sqrt() /(self.r_inter + self.r_outer)
            self.alpha = 1 - ( cos_a * cos_b -  sin_a * sin_b ) #  1 - cos(a+b)
            self.beta = 1 - cos_a   # 类内最大cos距离  1 - cos(a)  直径  #与ranked_loss 定义的参数稍有不同  margin和beta可以互换  margin = alpha - beta
        else:
            raise Exception("No this kind of DistanceType")

        self.center = nn.Parameter(self.center_buffer, requires_grad=False)


    def forward(self, feat, labels):  #__call__
        """
        Args:
          x: feature matrix with shape (batch_size, feat_dim).
          labels: ground truth labels with shape (batch_size).
        """
        #1.根据上次运行结果改变center
        if self.center_lock == 0:
            if torch.equal(self.center_buffer + self.d_center, self.center_buffer):
                self.center_lock = 1
            else:
                self.center_buffer.add_(self.d_center * 0.005)
                self.center = nn.Parameter(self.center_buffer, requires_grad=False)


        #self.r_outer_buffer.add_(self.d_r_outer * 0.005)
        #self.r_outer_buffer.clamp_(min=0.0)
        #self.r_outer = nn.Parameter(self.r_outer_buffer, requires_grad=False)
        #self.r_outer = self.r_outer_buffer

        #self.r_inter = self.r_outer * self.io_ratio   #这个球体的分类能力，不光与R和r的大小有关，还与空间维度有关。维度越高,呢个区分的类别越多
        self.cluster_paramters = {"center": self.center, "r_inter": self.r_inter, "r_outer": self.r_outer} #记录下本次求取loss的参数配置

        #2.计算annulus_loss
        #计算圆环内外圆的半径（^2）
        self.annulus_r = self.r_inter #.pow(2)   # /feat_dim    圆环内圆半径(^2)
        self.annulus_R = (self.r_inter + 2*self.r_outer) #.pow(2) # /feat_dim  圆环外圆半径(^2)
        #计算anchor 与 所有样本的总center之间的距离(^2)
        batch_size = feat.size(0)
        ac_distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                     torch.pow(self.center, 2).sum(dim=1, keepdim=True).expand(batch_size, 1)
        ac_distmat.addmm_(1, -2, feat, self.center.t())
        ac_distmat.sqrt_()
        #找到圆环小圆之内的点
        ac_dist_under_r_pos = torch.lt(ac_distmat, self.annulus_r)
        a_under_ar = ac_distmat[ac_dist_under_r_pos]
        annulus_loss_ur = torch.sum(self.annulus_r - a_under_ar)
        # 找到圆环大圆之外的点
        ac_dist_beyond_R_pos = torch.gt(ac_distmat,  self.annulus_R)
        a_beyond_aR = ac_distmat[ac_dist_beyond_R_pos]
        annulus_loss_bR = torch.sum(a_beyond_aR - self.annulus_R)
        #计算总损失，可以加weight
        annulus_loss = (annulus_loss_ur + annulus_loss_bR)/batch_size

        #3.计算ranked_loss损失
        center_feat = feat - self.center.expand(batch_size, self.feat_dim)
        if self.distance_type == "euclidean":
            dist_mat = euclidean_dist_rank(center_feat, center_feat)   #这个距离计算用feat就可以
        elif self.distance_type == "cos":
            dist_mat = cosine_dist_rank(center_feat, center_feat)
        ranked_loss = rank_loss(1-dist_mat, labels, self.alpha, self.beta, self.tval)

        #4.计算总损失
        total_loss = ranked_loss + annulus_loss

        #5.计算d_center和d_r
        unit_vector_direction = normalize_rank(center_feat)
        d_scalar = ac_distmat * (ac_dist_under_r_pos.float() + ac_dist_beyond_R_pos.float()) - (ac_dist_under_r_pos.float() * self.annulus_r + ac_dist_beyond_R_pos.float() * self.annulus_R)
        d_vector = unit_vector_direction * d_scalar
        self.d_center = d_vector.sum(dim=0)
        self.d_r_outer = torch.sum(d_scalar)

        return total_loss, self.cluster_paramters

