# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
"""
import re
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

#from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet import *
from .backbones.densenet import *
from .backbones.multi_bagnet import *
from .backbones.bagnet import *

from ptflops import get_model_complexity_info   #计算模型参数量和计算能力

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

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if isinstance(m.bias, nn.Parameter):
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    def __init__(self,  base_name, num_classes, preAct=True, fusionType="concat"):
        super(Baseline, self).__init__()

        self.num_classes = num_classes
        self.base_name = base_name

        # baselineOutputType 和 classifierType
        self.frameworkDict = {"f-c":("feature", "normal"), "pl-c":("pre-logit", "post"), "fl-n":("final-logit", "none")}
        self.BCType = "f-c"   #默认是该模式   baseline & classifier
        self.baseOutputType = self.frameworkDict[self.BCType][0]
        self.classifierType = self.frameworkDict[self.BCType][1]


        self.GradCAM = False

        # 1.Backbone
        if base_name == 'resnet18':
            self.in_planes = 512
            self.base = resnet18()
        elif base_name == 'resnet34':
            self.in_planes = 512
            self.base = resnet34()
        elif base_name == 'resnet50':
            self.base = resnet50()
        elif base_name == 'resnet101':
            self.base = resnet101()
        elif base_name == 'resnet152':
            self.base = resnet152()
        elif base_name == "densenet121":
            self.base = densenet121()
            self.in_planes = self.base.num_output_features
        # 以下为了与multi_bagnet比较所做的调整网络
        elif base_name == "bagnet":
            self.base = bagnet9()
            self.in_planes = 2048
        elif base_name == "resnetS224":
            self.base = resnetS224()
            self.in_planes = self.base.num_output_features
        elif base_name == "densenetS224":
            self.base = densenetS224()
            self.in_planes = self.base.num_output_features
        elif base_name == "mbagnet121":
            # 预设参数
            self.BCType = "fl-n"#"pl-c"    #"f-c", "pl-c", "fl-n"
            self.baseOutputType = self.frameworkDict[self.BCType][0]
            self.classifierType = self.frameworkDict[self.BCType][1]
            self.baseOutChannels = self.num_classes
            if self.BCType == "pl-c":
                self.baseOutChannels = 5  #可自设定，比如病灶种类

            self.rf_logits_hook = 1   #是否加入hook
            self.heatmapFlag = 0      #是否保存热点图

            # 初始化网络
            self.base = mbagnet121(preAct=preAct, fusionType=fusionType, reduction=1, outputType=self.baseOutputType, rf_logits_hook=self.rf_logits_hook, num_classes=self.baseOutChannels, complexity=0)
            self.in_planes = self.base.num_features


        # 2.以下是classifier的网络结构（3种）
        # （1）normal-classifier模式: backbone提供特征，classifier只是线性分类器，需要用gap处理
        if self.classifierType == "normal":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)
        #  (2)post-classifier模式: backbone提供的是logits，不需要gap，只需线性classifier即可
        elif self.classifierType == "post":
            self.finalClassifier = nn.Linear(self.baseOutChannels, self.num_classes)
            self.finalClassifier.apply(weights_init_classifier)
        #  (3)none模式: backbone自带分类器
        elif self.classifierType == "none":
            print("Backbone with classifier itself.")

        # 3.所有的hook操作（按理来说应该放在各自的baseline里）
        # GradCAM
        if self.GradCAM == 1:
            self.inter_output = None
            self.inter_gradient = None
            for module_name, module in self.base.features.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    print(module_name)
                    if module_name == "transition1.conv":
                        self.GradCAM_BN = torch.nn.BatchNorm2d(1)

                        module.register_forward_hook(self.forward_hook_fn)
                        module.register_backward_hook(self.backward_hook_fn)
                        break


    def forward_hook_fn(self, module, input, output):
        self.inter_output = output  #将输入图像的梯度获取

    def backward_hook_fn(self, module, grad_in, grad_out):
        self.inter_gradient = grad_out[0]  #将输入图像的梯度获取


    def transmitClassifierWeight(self):   #将线性分类器回传到base中
        if self.classifierType == "post":
            #self.base.baselineClassifierWeight = self.finalClassifier.weight
            #self.base.baselineClassifierBias = self.finalClassifier.bias
            self.base.overallClassifierWeight = torch.matmul(self.finalClassifier.weight, self.base.classifier.weight)
            self.base.num_classes = self.num_classes   #重置新的num-classes

    def transimitBatchDistribution(self, BD):
        self.base.batchDistribution = BD

    def forward(self, x):
        # 分为三种情况：1.base只提供特征 2.base输出logit但需后处理 3
        if self.classifierType == "normal":
            base_out = self.base(x)
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            final_logits = self.classifier(feat)
        elif self.classifierType == "post":
            logits = self.base(x)
            final_logits = self.finalClassifier(logits)
        elif self.classifierType == "none":
            final_logits = self.base(x)

        return final_logits   # 其他参数可以用model的成员变量来传递


    # 载入参数
    def load_param(self, loadChoice, model_path):
        param_dict = torch.load(model_path)
        b = self.base.state_dict()

        # for densenet 参数名有差异，需要先行调整
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(param_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                param_dict[new_key] = param_dict[key]
                del param_dict[key]

        if loadChoice == "Base":
            for i in param_dict:
                newi = i.replace("base.", "")
                if newi not in self.base.state_dict() or "classifier" in newi:
                    print(i)
                    #print(newi)
                    continue
                self.base.state_dict()[newi].copy_(param_dict[i])

        elif loadChoice == "Overall":
            for i in param_dict:
                if i not in self.state_dict():
                    if "classifier" in i:
                        basei = "base."+i
                        self.state_dict()[basei].copy_(param_dict[i])
                        continue
                    print(i)
                    continue
                self.state_dict()[i].copy_(param_dict[i])

        elif loadChoice == "Classifier":
            for i in param_dict:
                if i not in self.classifier.state_dict():
                    continue
                self.classifier.state_dict()[i].copy_(param_dict[i])

    # 计算网络参数量的方式（2种）
    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        return param_count

    def count_param2(model, input_shape=(3, 224, 224)):
        with torch.cuda.device(0):
            flops, params = get_model_complexity_info(model, input_shape, as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return ('{:<30}  {:<8}'.format('Computational complexity: ', flops)) + (
                '{:<30}  {:<8}'.format('Number of parameters: ', params))