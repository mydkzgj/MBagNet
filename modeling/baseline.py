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

from .classifiers.hierarchy_linear import *


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
    def __init__(self, base_name, classifier_name, num_classes, base_classifier_Type="f-c",
                 preAct=True, fusionType="concat",
                 hookType="none", segmentationType="none", seg_num_classes=1, segSupervisedType="none",
                 gcamSupervisedType="none", guidedBP=0,
                 maskedImgReloadType="none", preReload=0,
                 branch_img_num=0, branchConfigType="none",
                 accumulation_steps=1,
                 ):
        super(Baseline, self).__init__()
        # 0.参数预设
        self.num_classes = num_classes
        self.base_name = base_name
        self.classifier_name = classifier_name
        self.accumulation_steps = accumulation_steps

        # 用于处理mbagnet的模块类型
        self.preAct = preAct
        self.fusionType = fusionType

        # baselineOutputType 和 classifierType  "f-c" "pl-c" "fl-n"   网络搭建的三种模式
        self.frameworkDict = {"f-c":("feature", "normal"), "pl-c":("pre-logit", "post"), "fl-n":("final-logit", "none")}
        self.BCType = base_classifier_Type   #默认是该模式   baseline & classifier
        self.baseOutputType = self.frameworkDict[self.BCType][0]
        self.classifierType = self.frameworkDict[self.BCType][1]
        if self.BCType == "f-c":
            self.base_num_classes = 0
            self.base_with_classifier = False
        elif self.BCType == "pl-c":
            self.base_num_classes = seg_num_classes  # 可自设定，比如病灶种类
            self.base_with_classifier = True
        else:
            self.base_num_classes = self.num_classes
            self.base_with_classifier = True
            self.classifier_name = "none"


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

        self.guidedBP = guidedBP    # 是否采用导向反向传播计算梯度

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

        self.preReload = preReload   #reload是前置（与第一批同时送入）还是后置
        #self.gcamState = 1


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
        self.choose_backbone()

        # 2.以下是classifier的网络结构（3种）
        self.choose_classifier()

        # 3.所有的hook操作（按理来说应该放在各自的baseline里）
        self.set_hooks()


    def forward(self, x):
        if self.gcamState == True:
            self.inter_output.clear()
            self.inter_gradient.clear()

        # 分为两种情况：
        if self.classifier_name != "none":
            base_out = self.base(x)
            final_logits = self.classifier(base_out)
        else:
            final_logits = self.base(x)

        return final_logits   # 其他参数可以用model的成员变量来传递



    # Hook Function
    #1.保留中间输出——用于GradCAM
    def forward_hook_fn(self, module, input, output):
        self.inter_output.append(output)
    """
    def backward_hook_fn(self, module, grad_in, grad_out):
        if self.batchDistribution != 0:
            if self.batchDistribution != 1:
                self.inter_gradient.append(grad_out[0][self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]])
            else:
                # self.inter_gradient = grad_out[0]  #将输入图像的梯度获取
                self.inter_gradient.append(grad_out[0])  # 将输入图像的梯度获取
    """

    #2. 用于Guided Backpropgation
    #用于Relu处的hook
    def guided_backward_hook_fn(self, module, grad_in, grad_out):
        #self.gradients = grad_in[1]
        if self.guidedBPstate == True:
            pos_grad_out = grad_out[0].gt(0)
            result_grad = pos_grad_out * grad_in[0]
            #print(1)
            return (result_grad,)
        else:
            #print(2)
            pass

    #3.用于打印梯度
    def print_grad(self, module, grad_in, grad_out):
        if self.guidedBPstate == 0:
            print("start:")
            print("mean:{},  max:{},  min:{}".format(grad_out[0].abs().mean().item(), grad_out[0].max().item(), grad_out[0].min().item()))

    def gcamNormalization(self, gcam):
        # 归一化 v1 使用均值，方差
        """
        gcam_flatten = gcam.view(gcam.shape[0], -1)
        gcam_var = torch.var(gcam_flatten, dim=1).detach()
        gcam = gcam/gcam_var
        gcam = torch.sigmoid(gcam)
        #"""
        # 归一化 v2 正负分别用最大值归一化
        """        
        pos = torch.gt(gcam, 0).float()
        gcam_pos = gcam * pos
        gcam_neg = gcam * (1 - pos)
        gcam_pos_abs_max = torch.max(gcam_pos.view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
        gcam_neg_abs_max = torch.max(gcam_neg.abs().view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
            -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
        sigma = 1  # 0.8
        gcam = gcam_pos / (gcam_pos_abs_max.clamp(min=1E-12).detach() * sigma) + gcam_neg / gcam_neg_abs_max.clamp(
            min=1E-12).detach()  # [-1,+1]
        #"""
        # 归一化 v3 正负统一用绝对值最大值归一化
        gcam_abs_max = torch.max(gcam.abs().view(gcam.shape[0], -1), dim=1)[0]
        gcam_abs_max_expand = gcam_abs_max.clamp(1E-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
        gcam = gcam / (gcam_abs_max_expand.clamp(min=1E-12).detach())   # [-1,+1]
        #print("gcam_max{}".format(gcam_abs_max.mean().item()))
        return gcam, gcam_abs_max  #.mean().item()

        # 其他
        # gcam = torch.relu(torch.tanh(gcam))
        # gcam = gcam/gcam_abs_max
        # gcam_abs_max = torch.max(gcam.abs().view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(
        #    -1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)

        # gcam_pos_mean = (torch.sum(gcam_pos) / torch.sum(pos).clamp(min=1E-12))
        # gcam_neg_mean = (torch.sum(gcam_neg) / torch.sum(1-pos).clamp(min=1E-12))

        # gcam = (1 - torch.relu(-gcam_pos / (gcam_pos_abs_max.clamp(min=1E-12).detach() * sigma) + 1)) #+ gcam_neg / gcam_neg_abs_max.clamp(min=1E-12).detach()  # cjy
        # gcam = (1 - torch.relu(-gcam_pos / (gcam_pos_mean.clamp(min=1E-12).detach()) + 1)) + gcam_neg / gcam_neg_abs_max.clamp(min=1E-12).detach()
        # gcam = torch.tanh(gcam_pos/gcam_pos_mean.clamp(min=1E-12).detach()) + gcam_neg/gcam_neg_abs_max.clamp(min=1E-12).detach()
        # gcam = gcam_pos / gcam_pos_mean.clamp(min=1E-12).detach() #+ gcam_neg / gcam_neg_mean.clamp(min=1E-12).detach()
        # gcam = gcam/2 + 0.5

        # gcam_max = torch.max(torch.relu(gcam).view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
        # gcam_min = torch.min(gcam.view(gcam.shape[0], -1), dim=1)[0].clamp(1E-12).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(gcam)
        # gcam = torch.relu(gcam) / gcam_max.detach()

        # gcam = torch.tanh(gcam*4)
        # gcam = torch.relu(gcam)

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

                #lm = 0 * lm   #需要掩盖的病灶
            elif GradeLabel[i] == 2:
                lm1 = LesionMask[i:i + 1, 0:3]   # 改2为3
                lm2 = LesionMask[i:i + 1, 3:4]
                lm = torch.cat([lm1, lm2], dim=1)

                #lm = LesionMask[i:i + 1]    # 不区分病灶
            elif GradeLabel[i] == 3:
                lm = LesionMask[i:i + 1, 1:2]
                lm = 1 - lm * 0

            elif GradeLabel[i] == 4:
                lm = LesionMask[i:i + 1, 1:2]
                lm = 1 - lm * 0

            else:
                continue
            lm = torch.max(lm, dim=1, keepdim=True)[0]
            MaskList.append(lm)
        FusionMask = torch.cat(MaskList, dim=0)
        return FusionMask

    # 将与该疾病无关的病灶去除
    def lesionFusionForV1(self, LesionMask, GradeLabel):
        MaskList = []
        for i in range(GradeLabel.shape[0]):
            if GradeLabel[i] == 1:
                lm = LesionMask[i:i + 1, 2:3]

                max_kernel_size = 200
                lm = torch.nn.functional.max_pool2d(lm, kernel_size=max_kernel_size * 2 + 1, stride=1, padding=max_kernel_size)
                lm = 1 - lm
                #lm = 0 * lm   #需要掩盖的病灶
            elif GradeLabel[i] == 2:
                #lm1 = LesionMask[i:i + 1, 0:2]
                #lm2 = LesionMask[i:i + 1, 3:4]
                #lm = torch.cat([lm1, lm2], dim=1)

                #lm = LesionMask[i:i + 1]    # 不区分病灶

                lm = LesionMask[i:i + 1, 2:3]
            elif GradeLabel[i] == 3:
                #lm = LesionMask[i:i + 1, 1:2]
                #lm = 1 - lm * 0

                lm1 = LesionMask[i:i + 1, 0:1]
                lm2 = LesionMask[i:i + 1, 2:4]
                lm = torch.cat([lm1, lm2], dim=1)

            elif GradeLabel[i] == 4:
                #lm = LesionMask[i:i + 1, 1:2]
                #lm = 1 - lm * 0
                lm1 = LesionMask[i:i + 1, 0:1]
                lm2 = LesionMask[i:i + 1, 2:4]
                lm = torch.cat([lm1, lm2], dim=1)
            else:
                continue
            lm = torch.max(lm, dim=1, keepdim=True)[0]
            MaskList.append(lm)
        FusionMask = torch.cat(MaskList, dim=0)
        return FusionMask

    # 将grade1，2的病灶全标出来  3，4 由于无法标出所有就不标了
    def lesionFusionForV3(self, LesionMask, GradeLabel):
        MaskList = []
        for i in range(GradeLabel.shape[0]):
            if GradeLabel[i] == 1:
                lm = LesionMask[i:i + 1]

            elif GradeLabel[i] == 2:
                lm = LesionMask[i:i + 1]

            elif GradeLabel[i] == 3:
                lm = LesionMask[i:i + 1, 1:2]
                lm = 1 - lm * 0

            elif GradeLabel[i] == 4:
                lm = LesionMask[i:i + 1, 1:2]
                lm = 1 - lm * 0

            else:
                continue
            lm = torch.max(lm, dim=1, keepdim=True)[0]
            MaskList.append(lm)
        FusionMask = torch.cat(MaskList, dim=0)
        return FusionMask

    # 为了在train中固定BN  目前无用
    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.track_running_stats = False
            #m.eval()


    # choose backbone
    def choose_backbone(self):
        # 1.VGG
        if self.base_name == 'vgg16':
            self.base = vgg16(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
            self.in_planes = 512
        elif self.base_name == "vgg19":
            self.base = vgg19(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)

        # 2.ResNet
        elif self.base_name == 'resnet18':
            self.in_planes = 512
            self.base = resnet18(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
        elif self.base_name == 'resnet34':
            self.in_planes = 512
            self.base = resnet34(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
        elif self.base_name == 'resnet50':
            self.base = resnet50(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
        elif self.base_name == 'resnet101':
            self.base = resnet101(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
        elif self.base_name == 'resnet152':
            self.base = resnet152(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)

        # 3. DenseNet
        elif self.base_name == "densenet121":
            self.base = densenet121(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)  # densenet121o-640.yml
            """
            self.base = mbagnet121(num_classes=self.baseOutChannels,
                                   preAct=True, fusionType="concat", reduction=1, complexity=0,
                                   transitionType="non-linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType, seg_num_classes=self.seg_num_classes,
                                   )
            #"""
            self.in_planes = self.base.num_features
        elif self.base_name == "densenet201":
            self.base = densenet201(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
            """
            self.base = mbagnet201(num_classes=self.base_num_classes,
                                   preAct=True, fusionType="concat", reduction=1, complexity=0,
                                   transitionType="non-linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType,
                                   seg_num_classes=self.seg_num_classes,
                                   )
            # """
            self.in_planes = self.base.num_features

        # 以下为了与multi_bagnet比较所做的调整网络
        elif self.base_name == "bagnet":
            self.base = bagnet9()
            self.in_planes = 2048
        elif self.base_name == "resnetS224":
            self.base = resnetS224()
            self.in_planes = self.base.num_output_features
        elif self.base_name == "densenetS224":
            self.base = densenetS224()
            self.in_planes = self.base.num_features
        elif self.base_name == "mbagnet121":
            self.base = mbagnet121(num_classes=self.base_num_classes,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0,
                                   transitionType="linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType,
                                   seg_num_classes=self.seg_num_classes,
                                   )
            self.in_planes = self.base.num_features

        elif self.base_name == "mbagnet201":
            self.base = mbagnet201(num_classes=self.base_num_classes,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0,
                                   transitionType="linear",
                                   outputType=self.baseOutputType,
                                   hookType=self.hookType, segmentationType=self.segmentationType,
                                   seg_num_classes=self.seg_num_classes,
                                   )
            self.in_planes = self.base.num_features

    def choose_classifier(self):
        # （1）linear  （2）hierarchy_linear  （3）multi-layer  (4)none
        if self.classifier_name == "linear":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)
        elif self.classifier_name == "hierarchy_linear":
            self.classifier = HierarchyLinear(self.in_planes, self.num_classes)
        elif self.classifier_name == "none":
            print("Backbone with classifier itself.")

    def set_hooks(self):
        # 1.GradCAM hook               GradCAM 如果其不为none，那么就设置hook
        if self.gcamState == True:
            self.inter_output = []
            self.inter_gradient = []

            self.target_layer = [
                "denseblock4"]  # ["denseblock3"]#["denseblock1", "denseblock2", "denseblock3", "denseblock4"]#, "denseblock2", "denseblock3", "denseblock4"]#, "denseblock2", "denseblock3", "denseblock4"]#"conv0"#"denseblock3"#"conv0"#"denseblock1"  "denseblock2", "denseblock3",
            # "denseblock1", "denseblock2", "denseblock3",   ["denseblock4"]#  ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
            if self.target_layer != []:
                for tl in self.target_layer:
                    for module_name, module in self.base.features.named_modules():
                        # if isinstance(module, torch.nn.Conv2d):
                        if module_name == tl:  # "transition1.conv":
                            print("Grad-CAM hook on ", module_name)
                            module.register_forward_hook(self.forward_hook_fn)
                            # module.register_backward_hook(self.backward_hook_fn)  不以backward求取gcam了
                            break
        # 2.Guided Backpropagation Hook
        if self.guidedBP == True:
            print("Set GuidedBP Hook on Relu")
            for module_name, module in self.named_modules():
                if isinstance(module, torch.nn.ReLU) == True:
                    module.register_backward_hook(self.guided_backward_hook_fn)
        self.guidedBPstate = 0  # 用的时候再使用

        # 3.观测梯度 hook
        # 打印梯度
        self.need_print_grad = 0
        if self.need_print_grad == 1:
            self.target_layers = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
            if self.target_layers != []:
                for tl in self.target_layers:
                    for module_name, module in self.base.features.named_modules():
                        # if isinstance(module, torch.nn.Conv2d):
                        if module_name == tl:  # "transition1.conv":
                            print("Grad-CAM hook on ", module_name)
                            module.register_backward_hook(self.print_grad)  # 不以backward求取gcam了
                            break

    # 载入参数
    def load_param(self, loadChoice, model_path):
        param_dict = torch.load(model_path)
        b = self.base.state_dict()

        # For DenseNet 预训练模型与pytorch模型的参数名有差异，故需要先行调整
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
                module_name = i.replace("base.", "")
                if module_name not in self.base.state_dict():
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
                elif "fc" in module_name or "classifier" in module_name:
                    print("Donot load %s, have changed this module for retraining" % i)
                    continue
                self.base.state_dict()[module_name].copy_(param_dict[i])

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