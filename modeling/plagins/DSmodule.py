import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule

from torch.utils.tensorboard import SummaryWriter

from collections import OrderedDict

from mmcv.cnn import constant_init, kaiming_init

#做一个可视化
import cv2 as cv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def map_visualization(map_tensor):
    map = map_tensor.cpu().detach().numpy()
    show = map[0]
    #a = np.max(show)
    cv.imshow("1", show)
    cv.waitKey(0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-2], val=0)
    else:
        constant_init(m, val=0)

def weights_init_cjy(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class AttentionModule(nn.Module):
    """Attention module.

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
                 extension=2,
                 image_size=(300, 300),
                 layer_num=3,
                 base_kernel_size=3):
        super(AttentionModule, self).__init__()

        #设置4层吧
        self.layer_num = layer_num
        self.extension = extension
        self.in_channels = in_channels
        self.inter_channels = in_channels * self.extension
        self.base_kernel_size = base_kernel_size
        self.image_size = image_size

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            downsample_kernel_size = pow(self.base_kernel_size, layer_num - i - 1)
            stride = downsample_kernel_size//2
            layer = self._make_layer(self.in_channels, self.inter_channels, base_kernel_size=self.base_kernel_size, downsample_kernel_size=downsample_kernel_size, stride=stride, padding=0, dilation=1)
            self.layers.append(layer)

        self.upsample = nn.UpsamplingNearest2d(size=self.image_size)

        self.layerAttentionMaps = {}

        self.apply(weights_init_cjy)
        for i in range(self.layer_num):
            last_zero_init(self.layers[i])

        self.writer = SummaryWriter("D:/Research/MIP/Experiment/RLL/work_space/summary/attention")
        self.record_step = 0

        #self.init_weights()


    def _make_layer(self, in_channels, inter_channels, base_kernel_size=3, downsample_kernel_size=1, stride=1, padding=0, dilation=1):
        layer = nn.Sequential(OrderedDict([
            ('avg_pooling', nn.AvgPool2d(kernel_size=downsample_kernel_size, stride=downsample_kernel_size)),  #不重叠平均     降采样维原来的/downsample_kernel_size
            ('conv1', nn.Conv2d(in_channels, inter_channels, kernel_size=base_kernel_size, padding=base_kernel_size//2)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(inter_channels, 1, kernel_size=1)),
            ('sigmoid', nn.Sigmoid()),
        ]))
        return layer

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**-0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, input):
        batch_size, channels, height, width = input.shape

        for i in range(self.layer_num):
            if i == 0:
                self.currentLayerAttentionMap = self.layers[i](input)
                self.currentLayerAttentionMap = self.upsample(self.currentLayerAttentionMap)
            else:
                attentionedInput = torch.mul(input, self.currentLayerAttentionMap)
                newLayerAttentionMap = self.layers[i](attentionedInput)
                newLayerAttentionMap = self.upsample(newLayerAttentionMap)
                self.currentLayerAttentionMap = newLayerAttentionMap #torch.mul(self.currentLayerAttentionMap, newLayerAttentionMap)

            self.layerAttentionMaps[i] = self.currentLayerAttentionMap

        self.record_step = self.record_step + 1
        if self.record_step % 20 == 1:
            self.writer.add_images("Attention_AllBatch", self.currentLayerAttentionMap, self.record_step, dataformats='NCHW')
            for i in range(self.layer_num):
                self.writer.add_image("Attention"+str(i), self.layerAttentionMaps[i][0], self.record_step, dataformats='CHW')
                self.writer.flush()
                #map_visualization(self.layerAttentionMaps[i][0])


        #CJY 可视化
        """
        if self.in_channels == 256:
            context_mask = pairwise_weight[0][0].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][1000].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][1596].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][1995].view(h, w)
            map_visualization(context_mask)
        elif self.in_channels == 512:
            context_mask = pairwise_weight[0][0].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][290].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][406].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][560].view(h, w)
            map_visualization(context_mask)
        elif self.in_channels == 1024:
            context_mask = pairwise_weight[0][0].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][61].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][105].view(h, w)
            map_visualization(context_mask)
            context_mask = pairwise_weight[0][180].view(h, w)
            map_visualization(context_mask)
        #"""


        return self.currentLayerAttentionMap
