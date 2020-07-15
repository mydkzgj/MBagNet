import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from .utils import load_state_dict_from_url

from ..plagins.non_local import NonLocal2D
from ..plagins.context_block import ContextBlock

#from ..utils.orthogonal_conv import Conv2d
from torch.nn import Conv2d

from .densenet import *



model_urls = {
    'scnet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',  #densenet
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()

        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),   #改变norm
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient



    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

        self.out_channels = num_input_features + num_layers*growth_rate

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        out = torch.cat(features, 1)
        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



class PreNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12,),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False,):

        super(PreNet, self).__init__()

        self.num_classes = num_classes
        self.block_config = block_config
        self.num_layers = list(block_config)
        self.num_init_features = num_init_features
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

        # 记录关键模块的输出，主要是用于Fc-DenseNet
        self.key_features_channels_record = {}
        self.key_features_channels_record["first_layer_out"] = self.num_init_features


        # First convolution
        self.features = nn.Sequential()
        #"""
        self.features = nn.Sequential(OrderedDict([
            ('convN0', Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            #('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        #"""

        # Each denseblock
        self.num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=self.num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                drop_rate=self.drop_rate,
                memory_efficient=self.memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            self.num_features = self.num_features + num_layers * self.growth_rate
            if 1:#i != len(block_config) - 1:
                self.key_features_channels_record["transitionN%d" % (i + 1)] = self.num_features
                trans = _Transition(num_input_features=self.num_features,
                                    num_output_features=self.num_features // 2)
                self.features.add_module('transitionN%d' % (i + 1), trans)
                self.num_features = self.num_features // 2

        self.key_features_channels_record["final_output"] = self.num_features

        #CJY 原论文中没有最后的 norm 和 relu
        # Final batch norm
        self.features.add_module('normN5', nn.BatchNorm2d(self.num_features))

        # 2020.7.5 CJY 源代码中relu操作用在了forward中，那么就无法找到该模块。此处加进来
        self.features.add_module('relu5', nn.ReLU())

        # Linear layer
        # CJY at 2020.6.20
        self.pclassifier = nn.Conv2d(self.num_features, num_classes, kernel_size=1, stride=1, bias=False)


        # Official init from torch repo.
        for name, m in self.named_modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

        #self.calculateRF()


    def forward(self, x):
        #features = self.features(x)
        #out = F.relu(features, inplace=True)   #加入到了featrues里
        out = self.features(x)
        #out = F.adaptive_avg_pool2d(out, (1, 1))
        #out = torch.flatten(out, 1)
        out = self.pclassifier(out)
        return out


class SCNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, seg_num_classes=5, num_classes=2):

        super(SCNet, self).__init__()

        # 2个子网络
        self.SNet = PreNet(num_classes=seg_num_classes, block_config=(6,))
        self.Relay = nn.Conv2d(seg_num_classes, 3, kernel_size=3, stride=1, padding=1, bias=False)     #中继
        self.CNet = densenet121(num_classes=num_classes,)  #换成你自己的网络
        self.segmentation = None

    def forward(self, x):
        self.segmentation = self.SNet(x)
        input = self.Relay(torch.sigmoid(self.segmentation))
        out = self.CNet(input)
        return out


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    param_dict = load_state_dict_from_url(model_url, progress=progress)
    # For DenseNet 预训练模型与pytorch模型的参数名有差异，故需要先行调整
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
    for key in list(param_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            param_dict[new_key] = param_dict[key]
            del param_dict[key]

    #"""
    SNet_dict = model.SNet.state_dict()
    for i in param_dict:
        module_name = i.replace("base.", "")
        if module_name not in model.SNet.state_dict():
            print("Cannot load %s, Maybe you are using incorrect framework" % i)
            continue
        elif "fc" in module_name or "classifier" in module_name:
            print("Donot load %s, have changed this module for retraining" % i)
            continue
        model.SNet.state_dict()[module_name].copy_(param_dict[i])
    #"""

    CNet_dict = model.CNet.state_dict()
    for i in param_dict:
        module_name = i.replace("base.", "")
        if module_name not in model.CNet.state_dict():
            print("Cannot load %s, Maybe you are using incorrect framework" % i)
            continue
        elif "fc" in module_name or "classifier" in module_name:
            print("Donot load %s, have changed this module for retraining" % i)
            continue
        model.CNet.state_dict()[module_name].copy_(param_dict[i])

def _scnet(arch, seg_num_classes, num_classes, pretrained, progress,
              **kwargs):
    model = SCNet(seg_num_classes=seg_num_classes, num_classes=num_classes)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model

def scnet121(seg_num_classes=5, num_classes=2, pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _scnet('scnet121', seg_num_classes, num_classes, pretrained, progress,
                     **kwargs)