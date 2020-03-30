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
from .backbones.vgg import *

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
    def __init__(self, base_name, num_classes,
                 preAct=True, fusionType="concat",
                 base_classifier_Type="f-c", hierarchy_classifier=0,
                 hookType="none", segmentationType="none", seg_num_classes=1, segSupervisedType="none",
                 gcamSupervisedType="none",
                 maskedImgReloadType="none",
                 branch_img_num=0, branchConfigType="none",
                 accumulation_steps=1,
                 ):
        super(Baseline, self).__init__()
        # 0.参数预设
        self.num_classes = num_classes
        self.base_name = base_name
        self.accumulation_steps = accumulation_steps

        # 用于处理mbagnet的模块类型
        self.preAct = preAct
        self.fusionType = fusionType

        # baselineOutputType 和 classifierType  "f-c" "pl-c" "fl-n"
        self.frameworkDict = {"f-c":("feature", "normal"), "pl-c":("pre-logit", "post"), "fl-n":("final-logit", "none")}
        self.BCType = base_classifier_Type   #默认是该模式   baseline & classifier
        self.baseOutputType = self.frameworkDict[self.BCType][0]
        self.classifierType = self.frameworkDict[self.BCType][1]
        if self.BCType == "pl-c":
            self.baseOutChannels = seg_num_classes  # 可自设定，比如病灶种类
        else:
            self.baseOutChannels = self.num_classes


        # CJY 分类器是否采用hierachy的结构
        self.hierarchyClassifier = hierarchy_classifier

        # 下面为3个支路设置参数
        # bracnh used samples including seg, gracam, reload
        self.branch_img_num = branch_img_num
        # 用于记录实时的样本应用分布， 运行模型前可设置
        self.batchDistribution = 0

        # Branch1: hookType   "featureReserve":保存transition层features, "rflogitGenerate":生成rf_logit_map, "none"
        self.hookType = hookType
        # segType "denseFC", "none"
        self.segmentationType = segmentationType
        self.seg_num_classes = seg_num_classes
        self.segSupervisedType = segSupervisedType
        if self.segSupervisedType != "none":
            self.segState = True
        else:
            self.segState = False

        # Branch2: gradcamType
        #self.gradCAMType = gradcamType
        self.gcamSupervisedType = gcamSupervisedType
        if self.gcamSupervisedType != "none":
            self.gcamState = True
            if self.gcamSupervisedType == "seg_mask":
                self.segState = True
        else:
            self.gcamState = False

        # Branch3: MaskedImgReloadType  "none", "seg_mask", "gcam_mask"
        self.maskedImgReloadType = maskedImgReloadType
        if self.maskedImgReloadType != "none":
            self.reloadState = True
            if self.maskedImgReloadType == "seg_mask":
                self.segState = True
            elif self.maskedImgReloadType == "gcam_mask":
                self.gcamState = True
            elif self.maskedImgReloadType == "joint":
                self.segState = True
                self.gcamState = True
        else:
            self.reloadState = False


        """
        # Branch Config方式: 若其不为none，那么前面各支路参数的设置将会依据该项进行改写
        # "none", "weakSu-segRe", "strongSu-segRe", "jointSu-segRe", "strongSu-gcamRe", "noneSu-gcamRe",
        self.branchConfigType = branchConfigType
        if self.branchConfigType != "none":
            configList = self.branchConfigType.split("-")
            configList[0] = configList[0].replace("Su", "")
            configList[1] = configList[1].replace("Re", "")

            self.segSupervisedType = configList[0]
            if configList[0] == "weak":
                self.seg_num_classes = 1
                self.gradCAMType = "supervise_seg"
            elif configList[0] == "strong":
                self.seg_num_classes = 4
                self.gradCAMType = "none"
            elif configList[0] == "joint":
                self.seg_num_classes = 1 + 4
                self.gradCAMType = "supervise_seg"
            elif configList[0] == "none":
                self.seg_num_classes = 1 + 4
                self.gradCAMType = "none"
            else:
                raise Exception("Wrong Branch Config")

            if configList[1] == "seg":
                self.maskedImgReloadType = "seg_mask"
            elif configList[1] == "gcam":
                self.maskedImgReloadType = "gradcam_mask"
                self.gradCAMType = "reload"
            elif configList[1] == "joint":
                self.maskedImgReloadType = "joint_mask"
                self.gradCAMType = "reload"
            elif configList[1] == "none":
                self.maskedImgReloadType = "none"
            else:
                raise Exception("Wrong Branch Config")
        """


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
            #self.base = densenet121()
            #"""
            self.base = mbagnet121(num_classes=self.baseOutChannels,
                                   preAct=True, fusionType="concat", reduction=1, complexity=0,
                                   transitionType="non-linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType, seg_num_classes=self.seg_num_classes,
                                   )
            #"""
            self.in_planes = self.base.num_features
        elif base_name == "densenet201":
            # self.base = densenet201()
            # """
            self.base = mbagnet201(num_classes=self.baseOutChannels,
                                   preAct=True, fusionType="concat", reduction=1, complexity=0,
                                   transitionType="non-linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType,
                                   seg_num_classes=self.seg_num_classes,
                                   )
            # """
            self.in_planes = self.base.num_features

        # 以下为了与multi_bagnet比较所做的调整网络
        elif base_name == "bagnet":
            self.base = bagnet9()
            self.in_planes = 2048
        elif base_name == "resnetS224":
            self.base = resnetS224()
            self.in_planes = self.base.num_output_features
        elif base_name == "densenetS224":
            self.base = densenetS224()
            self.in_planes = self.base.num_features
        elif base_name == "mbagnet121":
            self.base = mbagnet121(num_classes=self.baseOutChannels,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0,
                                   transitionType="linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType, seg_num_classes=self.seg_num_classes,
                                   )
            self.in_planes = self.base.num_features

        elif base_name == "mbagnet201":
            self.base = mbagnet201(num_classes=self.baseOutChannels,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0,
                                   transitionType="linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType,
                                   seg_num_classes=self.seg_num_classes,
                                   )
            self.in_planes = self.base.num_features

        elif base_name == "vgg19": #(不要BN层)
            self.base = vgg19(pretrained=True)

        # 2.以下是classifier的网络结构（3种）
        # （1）normal-classifier模式: backbone提供特征，classifier只是线性分类器，需要用gap处理
        if self.classifierType == "normal":
            self.gap = nn.AdaptiveAvgPool2d(1)
            if self.hierarchyClassifier == False:
                self.classifier = nn.Linear(self.in_planes, self.num_classes)
                self.classifier.apply(weights_init_classifier)
            else:
                """
                self.classifier1 = nn.Linear(self.in_planes, 2)
                self.classifier1.apply(weights_init_classifier)
                self.classifier2 = nn.Linear(self.in_planes, 2)
                self.classifier2.apply(weights_init_classifier)
                self.classifier3 = nn.Linear(self.in_planes, 2)
                self.classifier3.apply(weights_init_classifier)
                self.classifier4 = nn.Linear(self.in_planes, 2)
                self.classifier4.apply(weights_init_classifier)
                self.classifier5 = nn.Linear(self.in_planes, 2)
                self.classifier5.apply(weights_init_classifier)
                self.classifier6 = nn.Linear(self.in_planes, 2)
                self.classifier6.apply(weights_init_classifier)
                #"""
                self.classifier1 = nn.Linear(self.in_planes, 1)
                self.classifier1.apply(weights_init_classifier)
                self.classifier2 = nn.Linear(self.in_planes, 1)
                self.classifier2.apply(weights_init_classifier)
                self.classifier3 = nn.Linear(self.in_planes, 1)
                self.classifier3.apply(weights_init_classifier)
                self.classifier4 = nn.Linear(self.in_planes, 1)
                self.classifier4.apply(weights_init_classifier)
                self.classifier5 = nn.Linear(self.in_planes, 1)
                self.classifier5.apply(weights_init_classifier)
                self.classifier6 = nn.Linear(self.in_planes, 1)
                self.classifier6.apply(weights_init_classifier)

        #  (2)post-classifier模式: backbone提供的是logits，不需要gap，只需线性classifier即可
        elif self.classifierType == "post":
            self.finalClassifier = nn.Linear(self.baseOutChannels, self.num_classes)
            self.finalClassifier.apply(weights_init_classifier)
        #  (3)none模式: backbone自带分类器
        elif self.classifierType == "none":
            print("Backbone with classifier itself.")

        # 3.所有的hook操作（按理来说应该放在各自的baseline里）
        # GradCAM 如果其不为none，那么就设置hook
        if self.gcamState == True:
            self.inter_output = [] #None
            self.inter_gradient = [] #None
            #self.INLayers = torch.nn.ModuleList()
            #self.projectors = torch.nn.Conv2d(1,1,kernel_size=1,bias=False)
            #nn.init.constant_(self.projectors.weight, 1)

            self.target_layer = ["denseblock4"]#"conv0"#"denseblock3"#"conv0"#"denseblock1"  "denseblock2", "denseblock3",
            #"denseblock1", "denseblock2", "denseblock3",
            if self.target_layer != []:
                for tl in self.target_layer:
                    #self.LNLayers.append(nn.InstanceNorm2d())
                    for module_name, module in self.base.features.named_modules():
                        #if isinstance(module, torch.nn.Conv2d):
                        if module_name == tl:  #"transition1.conv":
                            print("Grad-CAM hook on ", module_name)
                            #module.register_forward_hook(self.forward_hook_fn)
                            #module.register_backward_hook(self.backward_hook_fn)
                            break


    def forward(self, x):
        if self.gcamSupervisedType != "none":
            self.inter_output.clear()
            self.inter_gradient.clear()

        # 分为三种情况：1.base只提供特征 2.base输出logit但需后处理 3
        if self.classifierType == "normal":
            base_out = self.base(x)
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            if self.hierarchyClassifier == False:
                final_logits = self.classifier(feat)
            else:
                """
                logits1 = self.classifier1(feat)  # 画质（0，1，2，3，4，）Vs 5
                score1 = F.softmax(logits1, dim=-1)
                logits2 = self.classifier2(feat)  # 正常 0， （1，2，3，4）
                score2 = F.softmax(logits2, dim=-1)
                logits3 = self.classifier3(feat)  # 病灶 1, (2，3，4)
                score3 = F.softmax(logits3, dim=-1)
                logits4 = self.classifier4(feat)  # 病灶 2, (3，4)
                score4 = F.softmax(logits4, dim=-1)
                logits5 = self.classifier5(feat)  # 病灶 3, (4)
                score5 = F.softmax(logits5, dim=-1)

                score_c5 = score1[:, 1:2]
                score_L2 = score1[:, 0:1] * score2  # 0,(1,2,3,4)
                score_c0 = score_L2[:, 0:1]
                score_L3 = score_L2[:, 1:2] * score3  # 1,(2,3,4)
                score_c1 = score_L3[:, 0:1]
                score_L4 = score_L3[:, 1:2] * score4  # 2,(3,4)
                score_c2 = score_L4[:, 0:1]
                score_L5 = score_L4[:, 1:2] * score5  # 3,(4)

                final_logits = torch.cat([score_c0, score_c1, score_c2, score_L5, score_c5], dim=1)
                final_logits = torch.log(final_logits)
                #"""

                # 注：对于2分类问题，不应该用softmax，因为classifier会有2个超平面，而实际上只需要一个超平面
                logits1 = self.classifier1(feat)  # 画质（0，1，2，3，4，）Vs 5
                score1 = F.sigmoid(logits1)
                logits2 = self.classifier2(feat)  # 正常 0， （1，2，3，4）
                score2 = F.sigmoid(logits2)
                logits3 = self.classifier3(feat)  # 病灶 1, (2，3，4)
                score3 = F.sigmoid(logits3)
                logits4 = self.classifier4(feat)  # 病灶 2, (3，4)
                score4 = F.sigmoid(logits4)
                logits5 = self.classifier5(feat)  # 病灶 3, (4)
                score5 = F.sigmoid(logits5)

                score_c5 = score1
                score_L0 = 1-score1
                score_c0 = score_L0 * score2
                score_L1 = score_L0 * (1-score2)
                score_c1 = score_L1 * score3
                score_L2 = score_L1 * (1-score3)
                score_c2 = score_L2 * score4
                score_L3 = score_L2 * (1-score4)
                score_c3 = score_L3 * score5
                score_c4 = score_L3 * (1 - score5)

                final_logits = torch.cat([score_c0, score_c1, score_c2, score_c3, score_c4, score_c5], dim=1)
                #final_logits = torch.log(final_logits)  # 放到了loss里面，和nlll组合

        elif self.classifierType == "post":
            logits = self.base(x)
            final_logits = self.finalClassifier(logits)
        elif self.classifierType == "none":
            final_logits = self.base(x)

        return final_logits   # 其他参数可以用model的成员变量来传递


    def forward_hook_fn(self, module, input, output):
        if self.hierarchyClassifier == 0:
            if self.batchDistribution != 0:
                if self.batchDistribution != 1:
                    self.inter_output.append(
                        output[self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]])
                else:
                    # self.inter_output = output  #将输入图像的梯度获取
                    self.inter_output.append(output)  # 将输入图像的梯度获取
        else:
            self.inter_output.append(output)   #为了求其梯度，所以需要保存该模块输出的所有值


    def backward_hook_fn(self, module, grad_in, grad_out):
        if self.batchDistribution != 0:
            if self.batchDistribution != 1:
                self.inter_gradient.append(grad_out[0][self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]])
            else:
                # self.inter_gradient = grad_out[0]  #将输入图像的梯度获取
                self.inter_gradient.append(grad_out[0])  # 将输入图像的梯度获取


    def transmitClassifierWeight(self):   #将线性分类器回传到base中
        if self.segmentationType == "bagFeature" and self.hookType == "rflogitGenerate":
            if self.classifierType == "normal":
                self.base.overallClassifierWeight = self.classifier.weight
                self.base.num_classes = self.num_classes  # 重置新的num-classes
            elif self.classifierType == "post":
                # self.base.baselineClassifierWeight = self.finalClassifier.weight
                # self.base.baselineClassifierBias = self.finalClassifier.bias
                self.base.overallClassifierWeight = torch.matmul(self.finalClassifier.weight,
                                                                 self.base.classifier.weight)
                self.base.num_classes = self.num_classes  # 重置新的num-classes

    def transimitBatchDistribution(self, BD):
        self.batchDistribution = BD
        self.base.batchDistribution = BD

    def lesionFusion(self, LesionMask, GradeLabel):
        MaskList = []
        for i in range(GradeLabel.shape[0]):
            if GradeLabel[i] == 1:
                lm = LesionMask[i:i + 1, 2:3]
            elif GradeLabel[i] == 2:
                lm1 = LesionMask[i:i + 1, 0:2]
                lm2 = LesionMask[i:i + 1, 3:4]
                lm = torch.cat([lm1, lm2], dim=1)
                #sm = seg_mask[i:i + 1, 0:4]  #还是应该去除2
            elif GradeLabel[i] == 3:
                lm = LesionMask[i:i + 1, 1:2]
            elif GradeLabel[i] == 4:
                lm = LesionMask[i:i + 1, 1:2]
            else:
                continue
            lm = torch.max(lm, dim=1, keepdim=True)[0]
            MaskList.append(lm)
        FusionMask = torch.cat(MaskList, dim=0)
        return FusionMask


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
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
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