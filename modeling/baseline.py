# encoding: utf-8
"""
@author:  Jiayang Chen
@contact: sychenjiayang@163.com
"""
import re

from .backbones.resnet import *
from .backbones.densenet import *
from .backbones.multi_bagnet import *
from .backbones.bagnet import *
from .backbones.vgg import *
from .backbones.scnet import *

from .classifiers.hierarchy_linear import *

from .segmenters.fc_mbagnet import *

from .visualizers.grad_cam import *
from .visualizers.pgrad_cam import *
from .visualizers.grad_cam_plusplus import *
from .visualizers.backpropagation import *
from .visualizers.deconvolution import *
from .visualizers.guided_backpropagation import *
from .visualizers.guided_grad_cam import *
from .visualizers.pgrad_back_cam import *
from .visualizers.visual_backpropagation import *
from .visualizers.cjy import *
from .visualizers.guided_deconv_pgrad_cam import *


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
    def __init__(self, base_name,
                 classifier_name="linear", num_classes=6, base_classifier_Type="f-c",
                 segmenter_name="none", seg_num_classes=0,
                 visualizer_name="none", visual_target_layers="none",
                 preAct=True, fusionType="concat",
                 segSupervisedType="none",
                 gcamSupervisedType="none", guidedBP=0,
                 maskedImgReloadType="none", preReload=0,
                 branch_img_num=0, branchConfigType="none",
                 accumulation_steps=1,
                 ):
        super(Baseline, self).__init__()
        # 0.参数预设

        self.base_name = base_name
        self.classifier_name = classifier_name
        self.num_classes = num_classes
        self.segmenter_name = segmenter_name
        self.seg_num_classes = seg_num_classes
        self.visualizer_name = visualizer_name
        self.target_layer = visual_target_layers.split(" ") if visual_target_layers != "none" else []
        self.accumulation_steps = accumulation_steps

        # 用于处理mbagnet的模块类型
        self.preAct = preAct
        self.fusionType = fusionType

        # baselineOutputType 和 classifierType  "f-c" "pl-c" "fl-n"   网络搭建的三种模式
        self.frameworkDict = {"f-c":("feature", "normal"), "pl-c":("pre-logit", "post"), "fl-n":("final-logit", "none")}
        self.BCType = base_classifier_Type   #默认是该模式   baseline & classifier
        self.classifierType = self.frameworkDict[self.BCType][1]
        if self.BCType == "f-c":
            self.base_num_classes = 0
            self.base_with_classifier = False
        elif self.BCType == "pl-c":
            self.base_num_classes = self.seg_num_classes  # 可自设定，比如病灶种类
            self.base_with_classifier = True
        else:
            self.base_num_classes = self.num_classes
            self.base_with_classifier = True
            self.classifier_name = "none"

        # 下面为3个支路设置参数
        # bracnh used samples including seg, gracam, reload
        self.branch_img_num = branch_img_num
        # 用于记录实时的样本应用分布， 运行模型前可设置
        self.batchDistribution = 1

        # Branch1: segState
        # segType "denseFC", "none"
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

        # 1.Backbone
        self.choose_backbone()

        # 2.classifier的网络结构
        self.choose_classifier()

        # 3.classifier的网络结构
        self.segmenter = None
        self.segmentation = None
        self.choose_segmenter()

        # 4.visualizer
        # "grad-cam", "pgrad-cam-GBP", "pgrad-cam", "pgrad-cam-GBP", "grad-cam++", "grad-cam++-GBP",
        # "backpropagation", "deconvolution", "guided-backpropagation", "visual-backpropagation"
        # "guided-grad-cam","pgrad-back-cam","guided-deconv-pgrad-cam"
        self.visualizer_name = "none"#"guided-deconv-pgrad-cam"  #"guided-deconv-pgrad-cam" #"none" #"pgrad-cam"
        #self.target_layer = ["base.features.relu5"]
        #self.target_layer = ["base.features.1", "base.features.11", "base.features.20", "base.features.29", ""]
        #self.target_layer = ["base.features.relu0","base.features.denseblock1", "base.features.denseblock2", "base.features.denseblock3", "base.features.denseblock4", "base.features.relu5", ""]
        #self.target_layer = ["base.features."+str(i) for i in [1,3,6,20,29]]   #1,3,6,8,11,13,15,18,20,22,25,27,29


        #"""
        self.target_layer = []
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.ReLU) ) and "segmenter" not in module_name and "classifier" not in module_name:
                if "densenet" in self.base_name and "denseblock" not in module_name:
                    self.target_layer.append(module_name)
                elif "resnet" in self.base_name and "relu1" not in module_name and "relu2" not in module_name:
                    self.target_layer.append(module_name)
                elif "vgg" in self.base_name:
                    self.target_layer.append(module_name)
        #self.target_layer.append("")
        #"""


        self.visualizer = None
        self.visualization = None
        self.choose_visualizer()

        # 4.所有的hook操作（按理来说应该放在各自的baseline里）
        #self.set_hooks()


    def forward(self, x):
        if self.visualizer != None:
            x.requires_grad_(True)   #设置了这个，显存会随轮数增长，是因为无法释放吗?

        # 分为两种情况：
        if self.classifier_name != "none":
            base_out = self.base(x)
            final_logits = self.classifier(base_out)
        else:
            final_logits = self.base(x)

        if self.segmenter != None:
            if self.segmenter.batchDistribution != 0:  #self.segmenter.features_reserve != []:
                if self.segState == True:
                    self.segmentation = self.segmenter(self.segmenter.features_reserve[-1])
                else:
                    self.segmenter.features_reserve.clear()  # 如果不计算segmentation，那么就应该清除由hook保存的特征

        # for scnet
        if self.segmenter_name == "scnet":
            if self.batchDistribution != 0:
                if self.batchDistribution != 1:
                    self.segmentation = self.base.segmentation[self.batchDistribution[0]:self.batchDistribution[0] + self.batchDistribution[1]]
                else:
                    self.segmentation = self.base.segmentation

        return final_logits   # 其他参数可以用model的成员变量来传递


    # Hook Function
    #用于打印梯度
    def print_grad(self, module, grad_in, grad_out):
        if self.guidedBPstate == 0:
            print("start:")
            print("mean:{},  max:{},  min:{}".format(grad_out[0].abs().mean().item(), grad_out[0].max().item(), grad_out[0].min().item()))


    def transmitClassifierWeight(self):   #将线性分类器回传到base中
        if "mbagnet" in self.base_name:
            if self.BCType == "f-c":
                self.base.overallClassifierWeight = self.classifier.weight
                self.base.num_classes = self.num_classes  # 重置新的num-classes
            elif self.BCType == "pl-c":
                self.base.overallClassifierWeight = torch.matmul(self.finalClassifier.weight, self.base.classifier.weight)
                self.base.num_classes = self.num_classes  # 重置新的num-classes

    def transimitBatchDistribution(self, BD):
        self.batchDistribution = BD
        self.base.batchDistribution = BD
        if self.segmenter != None:
            self.segmenter.batchDistribution = BD

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

    # choose backbone
    def choose_backbone(self):
        # 1.VGG
        if self.base_name == 'vgg16':
            self.base = vgg16(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
            self.in_planes = 512
        elif self.base_name == "vgg19":
            self.base = vgg19(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
        elif self.base_name == "vgg16_bn":
            self.base = vgg16_bn(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)

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
            self.in_planes = self.base.num_features
        elif self.base_name == "densenet201":
            self.base = densenet201(num_classes=self.base_num_classes, with_classifier=self.base_with_classifier)
            self.in_planes = self.base.num_features

        # 4. MBagNet
        elif self.base_name == "mbagnet121":
            self.base = mbagnet121(num_classes=self.base_num_classes,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0, transitionType="linear",
                                   with_classifier=self.base_with_classifier,
                                   genarateLogitMapFlag=False,
                                   )
            self.in_planes = self.base.num_features

        elif self.base_name == "mbagnet201":
            self.base = mbagnet201(num_classes=self.base_num_classes,
                                   preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0, transitionType="linear",
                                   with_classifier=self.base_with_classifier,
                                   genarateLogitMapFlag=False,
                                   )
            self.in_planes = self.base.num_features

        # 5.scnet
        elif self.base_name == "scnet":
            self.base = scnet121(pretrained=True)
            self.in_planes = 1000

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
        else:
            raise Exception("Wrong Backbone Name!")

    def choose_classifier(self):
        # （1）linear  （2）hierarchy_linear  （3）multi-layer  (4)none
        if self.classifier_name == "linear":
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)
        elif self.classifier_name == "hierarchy_linear":
            self.classifier = HierarchyLinear(self.in_planes, self.num_classes)
        elif self.classifier_name == "none":
            print("Without Independent Classifier! (Backbone with classifier itself.)")


    def choose_segmenter(self):
        if self.segmenter_name == "fc_mbagnet":
            if "densenet" in self.base_name or "mbagnet" in self.base_name:
                self.segmenter = FCMBagNet(encoder=self.base, encoder_features_channels=self.base.key_features_channels_record,
                                           num_classes=self.seg_num_classes, batchDistribution=self.batchDistribution,
                                           growth_rate=self.base.growth_rate, block_config=self.base.block_config, bn_size=self.base.bn_size,
                                           preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0, transitionType="linear",
                                           )

        elif self.segmenter_name == "none":
            self.segmenter = None
            print("Without Segmenter!")

    def choose_visualizer(self):
        if self.visualizer_name == "grad-cam":
            self.visualizer = GradCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=False)
        elif self.visualizer_name == "grad-cam-GBP":
            self.visualizer = GradCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=True)
        elif self.visualizer_name == "pgrad-cam":  #pixel-wise grad-cam
            self.visualizer = PGradCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=False)
        elif self.visualizer_name == "pgrad-cam-GBP":  #pixel-wise grad-cam
            self.visualizer = PGradCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=True)
        elif self.visualizer_name == "grad-cam++":   #grad-cam++
            self.visualizer = GradCAMpp(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=False)
        elif self.visualizer_name == "grad-cam++-GBP":  # pixel-wise grad-cam
            self.visualizer = GradCAMpp(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=True)
        elif self.visualizer_name == "backpropagation":  #
            self.visualizer = Backpropagation(model=self, num_classes=self.num_classes)
        elif self.visualizer_name == "deconvolution":  #
            self.visualizer = Deconvolution(model=self, num_classes=self.num_classes)
        elif self.visualizer_name == "guided-backpropagation":  #
            self.visualizer = GuidedBackpropagation(model=self, num_classes=self.num_classes)
        elif self.visualizer_name == "visual-backpropagation":  #
            self.visualizer = VisualBackpropagation(model=self, num_classes=self.num_classes)
        elif self.visualizer_name == "guided-grad-cam":
            self.visualizer = GuidedGradCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer, useGuidedBP=True)
        elif self.visualizer_name == "pgrad-back-cam":
            self.visualizer = PGradBackCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer)
        elif self.visualizer_name == "cjy":
            self.visualizer = CJY(model=self, num_classes=self.num_classes, target_layer=self.target_layer)
        elif self.visualizer_name == "guided-deconv-pgrad-cam":
            self.visualizer = GuidedDeConvPGCAM(model=self, num_classes=self.num_classes, target_layer=self.target_layer)
        elif self.visualizer_name == "none":
            self.visualizer = None
            print("Without Visualizer!")





    def set_hooks(self):
        # 观测梯度 hook
        # 打印梯度
        self.need_print_grad = 0
        if self.need_print_grad == 1:
            self.target_layers = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
            if self.target_layers != []:
                for tl in self.target_layers:
                    for module_name, module in self.base.features.named_modules():
                        if module_name == tl:
                            print("Grad-CAM hook on ", module_name)
                            module.register_backward_hook(self.print_grad)
                            break

    # 载入参数
    def load_param(self, loadChoice, model_path):
        param_dict = torch.load(model_path)
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
            base_dict = self.base.state_dict()
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
            overall_dict = self.state_dict()
            for i in param_dict:
                if i in self.state_dict():
                    self.state_dict()[i].copy_(param_dict[i])
                elif "base."+i in self.state_dict():
                    self.state_dict()["base."+i].copy_(param_dict[i])
                else:
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue


        elif loadChoice == "Classifier":
            classifier_dict = self.classifier.state_dict()
            for i in param_dict:
                if i not in self.classifier.state_dict():
                    continue
                self.classifier.state_dict()[i].copy_(param_dict[i])

        print("Complete Load Weight")

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