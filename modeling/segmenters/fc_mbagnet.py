import re
import torch
import torch.nn as nn

from collections import OrderedDict
from .utils import load_state_dict_from_url

import modeling.backbones.multi_bagnet as MBN

import utils.featrueVisualization as fV


__all__ = ['FCMBagNet', 'fc_mbagnet121', 'fc_mbagnet169', 'fc_mbagnet201', 'fc_mbagnet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

#
class _TransitionUp(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionUp, self).__init__()
        #self.add_module('norm', nn.BatchNorm2d(num_input_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('tranconv', nn.ConvTranspose2d(num_input_features, num_output_features, kernel_size=2, stride=2, bias=False))


class FCMBagNet(nn.Module):
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

    def __init__(self, encoder, encoder_features_channels, num_classes, batchDistribution,
                 growth_rate=32, block_config=(6, 12, 24, 16),
                 bn_size=4, drop_rate=0, memory_efficient=False,
                 preAct=True, fusionType="concat", reduction=1, complexity=0, transitionType="linear",):
        super(FCMBagNet, self).__init__()

        # 依据 BackBone （DenseNet或MBagNet的配置进行对应的配置）
        #self.encoder = encoder
        self.decoder_features_channels = {}
        for i in encoder_features_channels.keys():
            if i == "final_output":
                self.decoder_features_channels["initial_input"] = encoder_features_channels["final_output"]
            elif "transition" in i:
                self.decoder_features_channels[i.replace("transition", "transitionUp")] = encoder_features_channels[i]
            elif i == "first_layer_out":
                self.decoder_features_channels["last_layer_out"] = encoder_features_channels["first_layer_out"]

        # Decoder Config
        self.decoder = nn.ModuleDict()
        self.num_classes = num_classes
        self.batchDistribution = batchDistribution
        self.num_layers = list(block_config)
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

        # MBagBlock配置
        self.preAct = preAct
        self.fusionType = fusionType
        self.reduction = reduction
        self.complexity = complexity

        # Transition配置
        self.transitionType = transitionType

        self.num_features = self.decoder_features_channels["initial_input"]

        # Each UpDenseblock
        for i, num_layers in enumerate(reversed(self.num_layers)):
            if i == 0:
                continue
            index = len(self.num_layers) - i  # 为了保证序号是对应的

            # transitionUp
            transUp = _TransitionUp(num_input_features=self.num_features,
                                    num_output_features=self.num_features // 2)  # 每个模块的输入为对应的transition层输出+前面的每个transitionUp的输出
            # self.segmenters.add_module('transitionUp%d' % (j + 1), transUp)
            self.decoder['transitionUp%d' % (index)] = transUp
            self.num_features = self.num_features // 2 + self.decoder_features_channels['transitionUp%d' % (index)]

            # blockUp
            self.seg_growth_rate = self.growth_rate  # //4
            blockUp = MBN._MBagBlock(
                num_layers=num_layers,
                num_input_features=self.num_features,
                bn_size=self.bn_size,
                growth_rate=self.seg_growth_rate,
                drop_rate=self.drop_rate,
                memory_efficient=self.memory_efficient,
                preAct=self.preAct,
                fusionType=self.fusionType,
                reduction=self.reduction,
                complexity=self.complexity
            )
            self.decoder['denseblockUp%d' % (index)] = blockUp
            self.num_features = blockUp.out_channels - self.num_features

        # lastLayer
        self.last_layer_num_features = self.decoder_features_channels["last_layer_out"] # //8  #缩减8倍
        self.decoder["last_layer"] = nn.Sequential(OrderedDict([
            ('tranconv0',
             nn.ConvTranspose2d(self.num_features, self.last_layer_num_features, kernel_size=3, stride=2, padding=1,
                                output_padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(self.last_layer_num_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('tranconv1',
             nn.ConvTranspose2d(self.last_layer_num_features, self.last_layer_num_features, kernel_size=7, stride=2, padding=3,
                                output_padding=1, bias=False)),
        ]))
        self.num_features = self.last_layer_num_features

        # descriminator
        self.descriminator = nn.Conv2d(self.num_features, self.num_classes, kernel_size=1, bias=False)

        # 设置为encoder网络设置hook，提取transition处的输出
        self.features_reserve = []
        modules = encoder.named_modules()
        for name, module in modules:
            """
            sub_name = name.split(".")
            if "denseblock" in sub_name[-1]:
                print("Set hook on {} for Fc-MBagNet".format(name))
                module.register_forward_hook(self.reserveFeature)
            #"""
            #"""
            # 下面这种模式，只是保留 avgpool模块前的输入  不等于 denseblock的输出
            if isinstance(module, nn.AvgPool2d) and "transition" in name:
                print("Set hook on {} for Fc-MBagNet".format(name))
                module.register_forward_hook(self.reserveFeature)
            elif isinstance(module, nn.AdaptiveAvgPool2d) and "gap" == name:
                print("Set hook on {} for Fc-MBagNet".format(name))
                module.register_forward_hook(self.reserveFeature)
            #"""

    def forward(self, features):
        for name in self.decoder.keys():
            # print(name)
            if "transitionUp" in name:
                transitionIndex = int(name.replace("transitionUp", ""))
                features = self.decoder[name](features)
                features = torch.cat([features, self.features_reserve[transitionIndex - 1]], dim=1)
            elif "denseblockUp" in name:
                input_channels = features.shape[1]
                features = self.decoder[name](features)
                output_channels = features.shape[1]
                features = features[:, input_channels:output_channels]
            elif "last_layer" in name:
                features = self.decoder[name](features)
        out = self.descriminator(features)
        self.features_reserve.clear()
        return out

    # forward hook function : 保存module的输出（部分or全部）
    def reserveFeature(self, module, input, output):
        #"""
        if self.batchDistribution != 0:
            if self.batchDistribution != 1:
                self.features_reserve.append(
                    input[0][self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]])
            else:
                self.features_reserve.append(input[0])
        #"""
        """
        if self.batchDistribution != 0:
            if self.batchDistribution != 1:
                self.features_reserve.append(
                    output[self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]])
            else:
                self.features_reserve.append(output)
        #"""

    # 显示分割结果
    def showDenseFCMask(self, seg_attention, imgs, labels, p_labels, masklabels=None, sample_index=0):
        img = imgs[sample_index]
        seg = torch.sigmoid(seg_attention[sample_index])
        if isinstance(masklabels, torch.Tensor):
            mask = masklabels[sample_index]
        else:
            mask =None
        fV.drawDenseFCMask(img, seg, mask)


def _load_state_dict(model, model_url, progress):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _fc_mbagnet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
             **kwargs):
    model = FCMBagNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model



def fc_mbagnet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _fc_mbagnet('mbagnet121', growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, pretrained=pretrained, progress=progress,
                    **kwargs)


def fc_mbagnet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _fc_mbagnet('mbagnet161', growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, pretrained=pretrained, progress=progress,
                    **kwargs)


def fc_mbagnet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _fc_mbagnet('mbagnet169', growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, pretrained=pretrained, progress=progress,
                    **kwargs)


def fc_mbagnet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _fc_mbagnet('mbagnet201', growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, pretrained=pretrained, progress=progress,
                    **kwargs)
