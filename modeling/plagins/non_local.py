import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule


#做一个可视化
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def map_visualization(map_tensor):
    map = map_tensor.cpu().detach().numpy()
    show = map
    a = np.max(show)
    #cv.imshow("1", show/a)
    #cv.waitKey(0)
    return show/a

def one_map_visualization(map_tensor):
    map = map_tensor.cpu().detach().numpy()
    show = map
    a = np.max(show)
    cv.imshow("1", show/a)
    cv.waitKey(0)
    return show/a

def maps_show(weight, h, w):
    # """h, w
    if 1:  # self.in_channels == 512:
        i = 0
        j = 0
        s = 4

        while i < h:
            while j < w:
                index = i * w + j
                print(i, j)
                context_mask = weight[0][index].view(h, w)
                map = map_visualization(context_mask)
                if j == 0:
                    hmap = map
                else:
                    hmap = cv.hconcat([hmap, map])
                j += s
            j = 0
            if i == 0:
                wholemap = hmap
            else:
                wholemap = cv.vconcat([wholemap, hmap])
            i += s
        # a = np.max(wholemap)
        cv.imshow("1", wholemap)
        cv.waitKey(0)
    # """

def maps_show2(weight, n):
    # """h, w
    if 1:  # self.in_channels == 512:
        batch_index = 0   #

        i = 0
        j = 0
        s = 1

        h = 8
        w = 8


        while i < h:
            while j < w:
                index = i * w + j
                if index >= n:
                    break
                print(i, j)
                context_mask = weight[index][0]
                map = map_visualization(context_mask)
                if j == 0:
                    hmap = map
                else:
                    hmap = cv.hconcat([hmap, map])
                j += s
            if index >= n:
                break
            j = 0
            if i == 0:
                wholemap = hmap
            else:
                wholemap = cv.vconcat([wholemap, hmap])
            i += s

        # a = np.max(wholemap)
        cv.imshow("batch_attention", wholemap)
        cv.waitKey(0)
    # """

class NonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 in_size,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.h = in_size[0]
        self.w = in_size[1]
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']


        #position_map  CJY
        #self.x_map = torch.arange(0, 1, 1/self.w).unsqueeze(dim=0).expand(self.h,self.w)
        #self.y_map = torch.arange(0, 1, 1/self.h).unsqueeze(dim=1).expand(self.h,self.w)
        #self.pos_map = torch.stack([self.x_map, self.y_map]).permute(1,2,0).cuda()

        #self.deta = nn.Parameter(torch.Tensor([0.01]).cuda(),  requires_grad=False)

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        #cjy 求聚类的权重
        self.global_map = ConvModule(
            self.in_channels,
            1,
            kernel_size=1,
            activation=None)

        #CJY pos-
        self.neighbor_out = ConvModule(
            self.inter_channels,
            1,
            kernel_size=1,
            activation=None)

        """
        #self.cluster_num = self.in_channels
        self.cluster_out = ConvModule(
            self.in_channels,
            1,
            kernel_size=1,
            activation=None)
        """
        self.cluster_out = nn.Sequential(
            #nn.Conv2d(self.in_channels, 16, kernel_size=1),
            #nn.Conv2d(16,1,kernel_size=1)
            ConvModule(self.in_channels, 16, kernel_size=1),
            ConvModule(16, 1, kernel_size=1, activation=None)
        )
        self.BN = nn.BatchNorm2d(1)


        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi, self.cluster_out[0]]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
            constant_init(self.global_map.conv, 0)   #CJY
            constant_init(self.neighbor_out.conv, 0)  # CJY
            constant_init(self.cluster_out[1].conv, 0)  # CJY
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x) * 10
        if self.use_scale:
            pass
            # theta_x.shape[-1] is `self.inter_channels`
            #pairwise_weight.div_(theta_x.shape[-1]**-0.5)    #除以这个值为1？是不是程序有问题

        # 变成cos距离，使得自己处的权值处于最大 为1
        """
        t_n = torch.norm(theta_x, p=2, dim=2)
        p_n = torch.norm(phi_x, p=2, dim=1)
        tp_n = torch.matmul(t_n.unsqueeze(2), p_n.unsqueeze(1))
        pairwise_weight /= tp_n
        """

        #cos-dist  #越发觉得这一点必加
        pairwise_weight.div_(torch.matmul(torch.norm(theta_x, p=2, dim=2).unsqueeze(2), torch.norm(phi_x, p=2, dim=1).unsqueeze(1)).clamp_(min=1e-12))

        total_weight = None
        p_weight = None
        """CJY 
        #pos-map   #用于开启位置的map      pos-map  *  strength-map
        #strength-map 由两个位置间的特征的关系决定
        #pos-map 应该由x,y来决定
        # H*W, 2
        pos_vector = self.pos_map.view(self.h * self.w, 2)
        # H*W, H*W, 2
        # abs_pos
        pos_weight = pos_vector.unsqueeze(0).expand(self.h * self.w, self.h * self.w, 2)

        # H*W, H*W
        # rel_pos  ^2
        re_pos_weight = torch.sum(torch.pow(pos_weight - pos_vector.unsqueeze(1), 2), dim=-1).sqrt_()

        # N, H*W, 1
        neighbor_deta = self.neighbor_out(phi_x.unsqueeze(-2))
        neighbor_deta.squeeze_(1).squeeze_(1).unsqueeze_(2)
        neighbor_deta.sigmoid_()
        neighbor_deta = 10 * (1-neighbor_deta)

        # H*W, H*W
        re_pos_weight = re_pos_weight.expand_as(pairwise_weight)
        p_weight = re_pos_weight * neighbor_deta

        p_weight.tanh_()
        p_weight = 1 - p_weight

        total_weight = p_weight + pairwise_weight  # 将特征与位置融合

        #以0为阈值进行筛选，求0以上元素的softmax
        total_weight.relu_()
        total_weight = total_weight.exp()
        total_weight = total_weight - 1

        #pairwise_weight = p_weight * pairwise_weight   #将特征与位置融合

        sum = total_weight.sum(dim=-1).unsqueeze(-1).expand_as(total_weight)
        total_weight = total_weight.div(sum)
        #"""

        pairwise_weight = pairwise_weight.softmax(dim=-1)
        #pairwise_weight.tanh_()


        return total_weight, pairwise_weight, p_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]

        #变成cos距离，使得自己处的权值处于最大 为1
        t_n = torch.norm(theta_x, p=2, dim=2)
        p_n = torch.norm(phi_x, p=2, dim=1)
        tp_n = t_n * p_n
        pairwise_weight /= tp_n

        return pairwise_weight

    """
    #原版 non-local
    def forward(self, x):
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)     #不利用g(x)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)   #将theta换成phi
        theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, HxW]
        phi_x = self.theta(x).view(n, self.inter_channels, -1)   #phi

        pairwise_func = getattr(self, self.mode)
        # total_weight: [N, HxW, HxW]
        total_weight, pairwise_weight, p_weight = pairwise_func(theta_x, phi_x)

        #maps_show(total_weight, h, w)
        #maps_show(pairwise_weight, h, w)
        #maps_show(p_weight, h, w)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        output = x + self.conv_out(y)

        cluster_out = self.cluster_out(output)
        cluster_BN = self.BN(cluster_out)
        attention_map = torch.relu(torch.tanh(cluster_BN)+0.5) * 2

        return output, attention_map

    """
    #cjy 修改
    def forward(self, x):
        n, _, h, w = x.shape


        # g_x: [N, HxW, C]
        g_x = x.view(n, self.in_channels, -1)  #x.view(n, self.in_channels, -1)  #self.g(x).view(n, self.inter_channels, -1)     #不利用g(x)
        g_x = g_x.permute(0, 2, 1)

        #CJY 将theta和phi算作一个函数
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        theta_x = phi_x.permute(0, 2, 1)

        pairwise_func = getattr(self, self.mode)
        # total_weight: [N, HxW, HxW]
        total_weight, pairwise_weight, p_weight = pairwise_func(theta_x, phi_x)

        #maps_show(total_weight, h, w)
        #maps_show(pairwise_weight, h, w)
        #maps_show(p_weight, h, w)        

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        #y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        #output = x + self.conv_out(y)

        output = y.permute(0, 2, 1).reshape(n, self.in_channels, h, w)    #配合去掉g
        cluster_out = self.cluster_out(output)
        cluster_BN = self.BN(cluster_out)
        attention_map = torch.relu(torch.tanh(cluster_BN)+0.5) * 2
        #c_o = self.cluster_out(output).view(n, 1, h*w)
        #cluster_output = torch.softmax(c_o, dim=-1).reshape(n, 1, h, w)
        #one_map_visualization(cluster_output[0][0])
        #one_map_visualization(attention_map[0][0])
        #one_map_visualization(c_o[0][0].view(h,w))
        #one_map_visualization(x[0][0])
        output = output * attention_map #* h * w

        # 通过新的特征进行全局Attention计算
        #global_map = torch.tanh(self.global_map(y))
        #y.mul_(global_map)

        #output = x + y

        #map_visualization(global_map[0][0])

        #maps_show2(attention_map, n)

        return output, attention_map
    #"""
