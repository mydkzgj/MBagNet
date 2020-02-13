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
from utils import featrueVisualization as fV



__all__ = ['MultiBagNet', 'mbagnetS224', 'mbagnet121', 'mbagnet169', 'mbagnet201', 'mbagnet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


#CJY 计算不同大小，不同位置的感受野提供的logits以及可视化



def _bn_function_factory(norm, relu, conv, preAct=True):
    if preAct == True:
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = conv(relu(norm(concated_features)))
            return bottleneck_output
    else:
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = relu(norm(conv(concated_features)))#conv(relu(norm(concated_features)))
            return bottleneck_output

    return bn_function

class _MBagLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False, preAct=True, reduction=1, complexity=0):
        super(_MBagLayer, self).__init__()

        self.preAct = preAct
        self.reduction = reduction
        self.complexity = complexity

        if self.preAct == True:
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
            #CJY at 2020.1.1  增加结构复杂度complexity这一参数，即增加多层1*1卷积
            if self.complexity > 0:
                self.add_module('meditation', torch.nn.Sequential())
            for i in range(self.complexity):
                self.meditation.add_module('norm'+str(i+3), nn.BatchNorm2d(growth_rate)),
                self.meditation.add_module('relu'+str(i+3), nn.ReLU(inplace=True)),
                self.meditation.add_module('conv'+str(i+3), Conv2d(growth_rate, growth_rate,
                                                                   kernel_size=1, stride=1, padding=0,
                                                                   bias=False)),

        else:
            self.add_module('conv1', Conv2d(num_input_features, bn_size *
                                            growth_rate, kernel_size=1, stride=1,
                                            bias=False)),
            self.add_module('norm1', nn.BatchNorm2d(bn_size * growth_rate)),  # 改变norm
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv2', Conv2d(bn_size * growth_rate, growth_rate,
                                            kernel_size=3, stride=1, padding=1,
                                            bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),

            # CJY at 2020.1.1  增加结构复杂度complexity这一参数，即增加多层1*1卷积
            if self.complexity > 0:
                self.add_module('meditation', torch.nn.Sequential())
            for i in range(self.complexity):
                self.meditation.add_module('conv' + str(i + 3), Conv2d(growth_rate, growth_rate,
                                                                       kernel_size=1, stride=1, padding=0,
                                                                       bias=False)),
                self.meditation.add_module('norm' + str(i + 3), nn.BatchNorm2d(growth_rate)),
                self.meditation.add_module('relu' + str(i + 3), nn.ReLU(inplace=True)),

        if reduction == 1:
            self.add_module('reduction1', nn.Identity())#nn.AvgPool2d(kernel_size=1, stride=1))
        elif reduction > 1:
            self.add_module('reduction1', Conv2d(growth_rate, growth_rate//reduction,
                                            kernel_size=1, stride=1, padding=0,
                                            bias=False)),


        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1, self.preAct)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        if self.preAct == True:
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        else:
            #new_features = self.relu5(self.norm5(self.conv5(self.relu4(self.norm4(self.conv4(self.relu3(self.norm3(self.conv3(self.relu2(self.norm2(self.conv2(bottleneck_output))))))))))))
            new_features = self.relu2(self.norm2(self.conv2(bottleneck_output)))

        #CJY meditation 冥想层
        if self.complexity > 0:
            new_features = self.meditation(new_features)

        if self.reduction == 1:    #如果有降维参数，那么可以提前降维
            new_features = self.reduction1(new_features)
        elif self.reduction > 1:
            new_features = self.reduction1(new_features)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features



class _MBagBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False, preAct=True, fusionType="concat", reduction=1, complexity=0):
        super(_MBagBlock, self).__init__()

        self.preAct = preAct
        self.fusionType = fusionType  # "add"  "concat" "none"
        self.reduction = reduction   #最后结果降维程度
        self.out_channels = num_input_features
        self.complexity = complexity

        if num_input_features != growth_rate//reduction and self.fusionType == "add":
            raise Exception("For add fusion type, num_input_features must equal with growth_rate!")

        for i in range(num_layers):
            layer = _MBagLayer(
                self.out_channels,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                preAct=self.preAct,
                reduction=self.reduction,
                complexity=self.complexity
            )
            if self.fusionType == "concat":
                self.out_channels = self.out_channels + growth_rate//self.reduction
            elif self.fusionType == "add":
                self.out_channels = self.out_channels + 0
            elif self.fusionType == "none":
                self.out_channels = self.out_channels + 0

            self.add_module('denselayer%d' % (i + 1), layer)


    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            if self.fusionType == "concat":
                features.append(new_features)
            elif self.fusionType == "add":
                features[0] = features[0] + new_features
            elif self.fusionType == "none":
                features[0] = new_features

        out = torch.cat(features, 1)
        return out


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, num_groups):
        super(_Transition, self).__init__()
        #self.add_module('norm', nn.BatchNorm2d(num_input_features))
        #self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', Conv2d(num_input_features, num_output_features,# groups=num_groups,  #cjy 新增groups=num_output_features
                                          kernel_size=1, stride=1, bias=False))
        # transition 主要负责降分辨率，同时也可负责感受野之间的加权降维（非group的conv）
        # 降低分辨率的方式：1.pool 2.conv(group)不打乱感受野
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



class MultiBagNet(nn.Module):
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

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=32, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False, preAct=True, fusionType="concat", transitionType="conv", classifierType="normal", rf_logits_hook=False, reduction=1, complexity=0):

        super(MultiBagNet, self).__init__()

        self.preAct = preAct
        self.fusionType = fusionType
        self.reduction = reduction
        self.complexity = complexity
        self.transitionType = transitionType
        self.classifierType = classifierType
        self.rf_logits_hook = rf_logits_hook

        self.num_classes = num_classes

        self.num_layers = list(block_config)
        self.num_init_features = num_init_features
        self.growth_rate = growth_rate

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.num_features = num_init_features
        self.num_receptive_field = 1

        if self.reduction == 1:
            self.features.add_module('reduction0', nn.Identity())#(kernel_size=1, stride=1))
        if self.reduction > 1:
            self.features.add_module('reduction0', Conv2d(self.num_features, self.num_features//reduction,
                                                         kernel_size=1, stride=1, padding=0,
                                                         bias=False))
            self.num_features = self.num_features//reduction

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _MBagBlock(
                num_layers=num_layers,
                num_input_features=self.num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                preAct=self.preAct,
                fusionType=self.fusionType,
                reduction=self.reduction,
                complexity=self.complexity
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            self.num_features = block.out_channels
            self.num_receptive_field = self.num_receptive_field + num_layers

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=self.num_features,
                                    num_output_features=self.num_features//2, num_groups=self.num_receptive_field)
                self.features.add_module('transition%d' % (i + 1), trans)
                self.num_features = self.num_features//2


        # CJY 原论文中没有最后的 norm 和 relu
        # Final batch norm
        #self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        #self.classifier = nn.Linear(num_features, num_classes)
        if self.classifierType == "normal":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        elif self.classifierType == "rfmode":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.rf_intra_classifier = nn.Conv2d(self.in_planes, self.num_receptive_field * self.num_classes,
                                                 kernel_size=1,  # groups=self.num_receptive_field,
                                                 stride=1, bias=False)
            self.rf_inter_classifier = nn.Linear(self.num_receptive_field, 1)
        elif self.classifierType == "none":
            print("Just pass features! No classifier")

        self.calculateRF()

        if rf_logits_hook == True:
            self.setHook(self.GenerateRFlogitMap)
        self.rf_logits_reserve = []
        self.rf_logits_reserve2 = []

        # Official init from torch repo.
        for name, m in self.named_modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                #nn.init.constant_(m.bias, 0)


    def calculateRF(self):
        # 记录下每个感受野的size，stride等参数
        self.receptive_field_list = []
        rf_size = 1
        rf_stride = 1
        rf_padding = 0
        modules = self.named_modules()
        for name, module in modules:
            if isinstance(module, nn.Conv2d):
                if "conv0" in name or "conv2" in name:
                    kernel_size = module.kernel_size[0]
                    stride = module.stride[0]
                    padding = module.padding[0]
                    rf_size = kernel_size * rf_stride + rf_size - rf_stride
                    rf_padding = padding * rf_stride + rf_padding
                    rf_stride = stride * rf_stride
                    self.receptive_field_list.append({"rf_size": rf_size, "rf_stride": rf_stride, "padding": rf_padding})
                    #print(name)
                    #print({"rf_size": rf_size, "rf_stride": rf_stride, "padding": rf_padding})
            if isinstance(module, nn.MaxPool2d):
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                rf_size = kernel_size * rf_stride + rf_size - rf_stride
                rf_padding = padding * rf_stride + rf_padding
                rf_stride = stride * rf_stride
                self.receptive_field_list[-1] = {"rf_size": rf_size, "rf_stride": rf_stride, "padding": rf_padding}  #MaxPool 要修改感受野大小

            if isinstance(module, nn.AvgPool2d):
                kernel_size = module.kernel_size
                stride = module.stride
                padding = module.padding
                rf_size = kernel_size * rf_stride + rf_size - rf_stride
                rf_padding = padding * rf_stride + rf_padding
                rf_stride = stride * rf_stride
        #print(self.receptive_field_list)

    #'''
    def GenerateRFlogitMap(self, module, input, output):
        #CJY 保存原始输出特征
        """
        self.hookType = "original_feature"
        if self.hookType == "original_feature":
            rf_logits = nn.functional.adaptive_avg_pool2d(output, 7)
            rf_logits = rf_logits.view(rf_logits.shape[0], rf_logits.shape[1], -1)
            #a = torch.norm(rf_logits, p=2, dim=-1, keepdim=True).expand_as(rf_logits)
            rf_logits = rf_logits/(torch.norm(rf_logits, p=2, dim=-1, keepdim=True).expand_as(rf_logits) + 1e-12)
            rf_logits = torch.mean(rf_logits, dim=1, keepdim=True)
            self.rf_logits_reserve2.append(rf_logits)
        #"""

        #"""
        layer_config = self.num_layers.copy()
        layer_config[0] = layer_config[0] + 1   #将conv0,maxpooling层合并，相当于只以transition层分割

        transitionLayerWeightList = []
        BlockStartPos = [0]
        for name, parameters in self.features.named_parameters():
            if "transition" in name and "conv" in name and "weight" in name:
                #print(name)
                transitionLayerWeightList.append(parameters)
                BlockStartPos.append(parameters.shape[0])
        if self.classifierType == "normal":  #将最终的线性层也归入到transitionLayer中，其实本质确是相同的
            transitionLayerWeightList.append(self.classifier.weight.unsqueeze(-1).unsqueeze(-1))
            BlockStartPos.append(self.classifier.weight.shape[0])

        # 确定module的位置
        #print(self.currentBlockIndex)
        #print(self.currentLayerIndex)

        if self.currentBlockIndex == 0:
            if self.currentLayerIndex == 0:
                weight_start_pos = BlockStartPos[self.currentBlockIndex]
                weight_end_pos = self.num_init_features
            else:
                weight_start_pos = self.num_init_features + (self.currentLayerIndex-1) * self.growth_rate
                weight_end_pos = weight_start_pos + self.growth_rate
        else:
            weight_start_pos = BlockStartPos[self.currentBlockIndex] + self.currentLayerIndex * self.growth_rate
            weight_end_pos = weight_start_pos + self.growth_rate

        #print(weight_start_pos)
        #print(weight_end_pos)

        #计算经过transition层的映射
        transitionLayerNum = len(self.num_layers)
        for i in range(transitionLayerNum):
            if i >= self.currentBlockIndex:
                transitionLayerWeight = transitionLayerWeightList[i][:,weight_start_pos:weight_end_pos]
                output = F.conv2d(output, transitionLayerWeight)
                weight_start_pos = 0
                weight_end_pos = transitionLayerWeight.shape[0]

        #计算经过
        # CJY at 2020.1.2  只取正值
        #output = torch.relu(output)
        self.rf_logits_reserve.append(output)
        


        self.currentLayerIndex = self.currentLayerIndex + 1
        if self.currentLayerIndex == layer_config[self.currentBlockIndex]:
            self.currentBlockIndex = self.currentBlockIndex + 1
            self.currentLayerIndex = 0
        #"""




    def setHook(self, hook_fn_forward):
        modules = self.named_modules()
        for name, module in modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Identity):
                if "reduction0" in name:
                    module.register_forward_hook(hook_fn_forward)
                if "reduction1" in name:
                    module.register_forward_hook(hook_fn_forward)


    # CJY
    def generateScoreMap(self, show_maps, rank_num_per_class=10):
        # CJY at 2019.12.5  计算不同感受野的图像的特征
        # 1.找到logits最大的几个evidence
        # (1)将特征logits拉平连接在一起
        for i in range(3, len(show_maps)-1):
            rf_intra_logit = show_maps[i]
            if i == 3:
                rf_intra_logit_flatten = rf_intra_logit.view(rf_intra_logit.shape[0], rf_intra_logit.shape[1], -1)
            else:
                rf_intra_logit_flatten = torch.cat([rf_intra_logit_flatten,
                                                    rf_intra_logit.view(rf_intra_logit.shape[0],
                                                                        rf_intra_logit.shape[1], -1)], dim=2)

            self.receptive_field_list[i - 3]["width"] = rf_intra_logit.shape[-1]
            self.receptive_field_list[i - 3]["size"] = rf_intra_logit.shape[-1] * rf_intra_logit.shape[-1]

        # (2)排序
        # 是否应该依据差值来求
        for c in range(self.num_classes):
            diff_rf_intra_logit_flatten = rf_intra_logit_flatten[0][c] - rf_intra_logit_flatten
            diff_rf_intra_logit_flatten = torch.sum(diff_rf_intra_logit_flatten, dim=1, keepdim=True)
            if c == 0:
                d_rf_intra_logit_flatten = diff_rf_intra_logit_flatten
            else:
                d_rf_intra_logit_flatten = torch.cat([d_rf_intra_logit_flatten, diff_rf_intra_logit_flatten], dim=1)
        rf_intra_logit_flatten = d_rf_intra_logit_flatten

        rank_logits = torch.sort(rf_intra_logit_flatten, dim=2, descending=True)  # 返回一个tuple  包含value和index

        pick_logits = rank_logits[0][0][:, 0:rank_num_per_class]
        pick_logits_index = rank_logits[1][0][:, 0:rank_num_per_class]

        # (3)依据index定位感受野层index和map中的（i，j）     #感受野大小，位置
        pick_logits_dict = {}
        for i in range(self.num_classes):
            pick_logits_dict[i] = []
            for j in range(rank_num_per_class):
                index = pick_logits_index[i][j].item()
                for rf_i in range(len(self.receptive_field_list)):  # 遍历所有感受野信息
                    if index - self.receptive_field_list[rf_i]["size"] < 0:  # 如果index<size，说明在在该感受野内，那么继续求取横纵坐标
                        h_index = index // self.receptive_field_list[rf_i]["width"]
                        w_index = index % self.receptive_field_list[rf_i]["width"]
                        padding = self.receptive_field_list[rf_i]["padding"]
                        rf_size = self.receptive_field_list[rf_i]["rf_size"]
                        rf_stride = self.receptive_field_list[rf_i]["rf_stride"]
                        center_x = -padding + rf_size // 2 + 1 + w_index * rf_stride
                        center_y = -padding + rf_size // 2 + 1 + h_index * rf_stride
                        break
                    else:
                        index -= self.receptive_field_list[rf_i]["size"]

                pick_logits_dict[i].append(
                    {"h": h_index, "w": w_index, "padding": padding, "rf_size": rf_size, "center_x": center_x,
                     "center_y": center_y, "logit": pick_logits[i][j].item(),
                     "max_padding": self.receptive_field_list[-1]["padding"]})

        # 2. 可视化
        fV.showrfFeatureMap(show_maps, self.num_classes, pick_logits_dict, AvgFlag=1)

    def forward(self, x):
        self.currentBlockIndex = 0
        self.currentLayerIndex = 0
        self.rf_logits_reserve.clear()
        #self.rf_logits_reserve2.clear()
        #求最后的特征输出
        features = self.features(x)
        #求最后的logits
        if self.classifierType == "none":  #不包含分类器，输出特征
            return features
        elif self.classifierType == "normal":  #包含分类器，输出logits
            if self.rf_logits_hook == True:
                #overall_rf_logits = F.conv2d(features, self.classifier.weight.unsqueeze(-1).unsqueeze(-1))
                #overall_rf_logits = torch.zeros_like(overall_rf_logits)

                # CJY at 2020.1.7   将上述的图进行（上采样）融合
                rf_list = []
                Max_FeatureMap_Scale = (self.rf_logits_reserve[0].shape[2], self.rf_logits_reserve[0].shape[3])  #(56, 56)
                for rlr in self.rf_logits_reserve:
                    #alpha = Max_FeatureMap_Scale[0]//rlr.shape[-1]
                    rlr_scale = torch.nn.functional.upsample_nearest(rlr, size=Max_FeatureMap_Scale, )#/(alpha*alpha)  #scale_factor=2
                    rf_list.append(rlr_scale.unsqueeze(0))
                    r = torch.cat(rf_list, dim=0)
                overall_rf_logits = torch.sum(r, dim=0)
                r1= torch.mean(torch.mean(torch.relu(overall_rf_logits), dim=-1), dim=-1)# + self.classifier.bias

                self.rf_logits_reserve.clear()
                self.rf_logits_reserve.append(overall_rf_logits)
                #self.rf_logits_reserve2.append(overall_rf_logits)
                #final_logits = self.gap(overall_rf_logits)

            #global_feat = self.gap(features)  # (b, ?, 1, 1)
            #feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            #final_logits = self.classifier(feat)
            return r1#final_logits



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


def _mbagnet(arch, growth_rate, block_config, num_init_features, pretrained, progress,
             **kwargs):
    model = MultiBagNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def mbagnetS224(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _mbagnet('mbagnetS224', 32, (6, 8, 8,), 32, pretrained, progress,
                    **kwargs)
    #return _mbagnet('mbagnet224', 32, (6, 8, 8), 32, pretrained, progress,
    #                **kwargs)

def mbagnet121(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _mbagnet('mbagnet121', 32, (6, 12, 24, 16), 64, pretrained, progress,   #(6, 12, 24, 16)
                    **kwargs)


def mbagnet161(pretrained=False, progress=True, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _mbagnet('mbagnet161', 48, (6, 12, 36, 24), 96, pretrained, progress,
                    **kwargs)


def mbagnet169(pretrained=False, progress=True, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _mbagnet('mbagnet169', 32, (6, 12, 32, 32), 64, pretrained, progress,
                    **kwargs)


def mbagnet201(pretrained=False, progress=True, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _mbagnet('mbagnet201', 32, (6, 12, 48, 32), 64, pretrained, progress,
                    **kwargs)
