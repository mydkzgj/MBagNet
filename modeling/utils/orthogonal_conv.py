#CJY 正交滤波器组

import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_conv_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    # 将weight转化为正交weight
    c_out = weight.size(0)
    weight_flat = weight.view(c_out, -1).unsqueeze(1)
    """
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_out, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_out, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    """

    for i in range(c_out):
        if i == 0:
            orth_weight_flat = weight_flat[i] / torch.norm(weight_flat[i], 2, -1, keepdim=True)
        else:
            temp = torch.matmul(weight_flat[i], orth_weight_flat.transpose(1, 0))
            temp = torch.matmul(temp, orth_weight_flat)
            temp = weight_flat[i] - temp
            orth_weight_flat = torch.cat([orth_weight_flat, temp / torch.norm(temp, 2, -1, keepdim=True)], dim=0)

    orth_weight = orth_weight_flat.view_as(weight)

    """
    #可视化
    y = F.conv2d(input, orth_weight, bias, stride, padding, dilation, groups)
    y_relu = F.relu(y)
    from utils import featrueVisualization as fV
    fV.maps_show3(y, y.shape[1])
    fV.maps_show3(y_relu, y_relu.shape[1])
    """

    return F.conv2d(input, orth_weight, bias, stride, padding, dilation, groups)


class Conv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return orthogonal_conv_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)
