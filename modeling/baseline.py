# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
"""
import re

from .backbones.resnet import *
from .backbones.densenet import *
from .backbones.multi_bagnet import *
from .backbones.bagnet import *
from .backbones.vgg import *

from .classifiers.hierarchy_linear import *

from .segmenters.fc_mbagnet import *


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
        self.choose_segmenter()
        self.segmentation = None

        self.visualization = None

        # 4.所有的hook操作（按理来说应该放在各自的baseline里）
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

        if self.segmenter_name != "none":
            if self.segState == True:
                self.segmentation = self.segmenter(self.segmenter.features_reserve[-1])
            else:
                self.segmenter.features_reserve.clear()   #如果不计算segmentation，那么就应该清除由hook保存的特征

        return final_logits   # 其他参数可以用model的成员变量来传递



    # Hook Function
    #1.保留中间输出——用于GradCAM
    def forward_hook_fn(self, module, input, output):
        self.inter_output.append(output)
    """
    # 最好不要在这里去切分输出的特征，因为后续如果要对其求梯度是不行的，必须先保留整体
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
        if self.guidedBPstate == True:
            pos_grad_out = grad_out[0].gt(0)
            result_grad = pos_grad_out * grad_in[0]
            return (result_grad,)
        else:
            pass

    #3.用于打印梯度
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
            print("Backbone with classifier itself.")


    def choose_segmenter(self):
        if self.segmenter_name == "fc_mbagnet":
            if "densenet" in self.base_name or "mbagnet" in self.base_name:
                self.segmenter = FCMBagNet(encoder=self.base, encoder_features_channels=self.base.key_features_channels_record,
                                           num_classes=self.seg_num_classes, batchDistribution=self.batchDistribution,
                                           growth_rate=self.base.growth_rate//4, block_config=self.base.block_config, bn_size=self.base.bn_size,
                                           preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0, transitionType="linear",
                                           )

    def choose_visualizer(self):
        if self.visualizer_name == "grad-cam":
            if "densenet" in self.base_name or "mbagnet" in self.base_name:
                self.visualizer = FCMBagNet(encoder=self.base, encoder_features_channels=self.base.key_features_channels_record,
                                           num_classes=self.seg_num_classes, batchDistribution=self.batchDistribution,
                                           growth_rate=self.base.growth_rate, block_config=self.base.block_config, bn_size=self.base.bn_size,
                                           preAct=self.preAct, fusionType=self.fusionType, reduction=1, complexity=0, transitionType="linear",
                                           )


    def generateGCAM(self, logits, labels, gcamBatchDistribution, device):
        # 将label转为one - hot
        gcam_one_hot_labels = torch.nn.functional.one_hot(labels, self.num_classes).float()
        gcam_one_hot_labels = gcam_one_hot_labels.to(device) if torch.cuda.device_count() >= 1 else gcam_one_hot_labels

        # 回传one-hot向量  已弃用 由于其会对各变量生成梯度，而使用op.zero_grad 或model.zero_grad 都会使程序出现问题，故改用torch.autograd.grad
        # logits.backward(gradient=one_hot_labels, retain_graph=True)#, create_graph=True)  #这样会对所有w求取梯度，且建立回传图会很大

        # 求取model.inter_output对应的gradient
        # 回传one-hot向量, 可直接传入想要获取梯度的inputs列表，返回也是列表
        self.guidedBPstate = 1  # 是否开启guidedBP
        inter_gradients = torch.autograd.grad(outputs=logits, inputs=self.inter_output,
                                              grad_outputs=gcam_one_hot_labels,
                                              retain_graph=True)  # , create_graph=True)
        self.inter_gradient = list(inter_gradients)
        self.guidedBPstate = 0

        # 生成CAM
        target_layer_num = len(self.target_layer)
        gcam_list = []
        gcam_max_list = []  # 记录每个Grad-CAM的归一化最大值
        for i in range(target_layer_num):
            gcam_max_list.append(1)

        for i in range(target_layer_num):
            inter_output = self.inter_output[i][
                           self.inter_output[i].shape[0] - gcamBatchDistribution[1]:self.inter_output[i].shape[0]]  # 此处分离节点，别人皆不分离  .detach()
            inter_gradient = self.inter_gradient[i][
                             self.inter_gradient[i].shape[0] - gcamBatchDistribution[1]:self.inter_gradient[i].shape[0]]
            if False:  # model.target_layer[i] == "denseblock4":  # 最后一层是denseblock4的输出，使用forward形式
                gcam = F.conv2d(inter_output, model.classifier.weight.unsqueeze(-1).unsqueeze(-1))
                # gcam = gcam /(gcam.shape[-1]*gcam.shape[-2])  #如此，形式上与其他层计算的gcam量级就相同了
                # gcam = torch.softmax(gcam, dim=-1)
                pick_label = labels[labels.shape[0] - gcamBatchDistribution[1]:labels.shape[0]]
                pick_list = []
                for j in range(pick_label.shape[0]):
                    pick_list.append(gcam[j, pick_label[j]].unsqueeze(0).unsqueeze(0))
                gcam = torch.cat(pick_list, dim=0)
            else:  # backward形式
                gcam = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
                gcam = gcam * (
                            gcam.shape[-1] * gcam.shape[-2])  # 如此，形式上与最后一层计算的gcam量级就相同了  （由于最后loss使用mean，所以此处就不mean了）
                gcam = torch.relu(gcam)  # CJY at 2020.4.18

            # print(gcam.sum(), gcam.mean(), gcam.abs().max())
            gcam = torch.relu(gcam)
            norm_gcam, gcam_max = self.gcamNormalization(gcam)

            # 插值
            # gcam = torch.nn.functional.interpolate(gcam, (seg_gt_masks.shape[-2], seg_gt_masks.shape[-1]), mode='bilinear')  #mode='nearest'  'bilinear'
            gcam_list.append(norm_gcam)  # 将不同模块的gcam保存到gcam_list中
            gcam_max_list[i] = gcam_max.detach().mean().item() / 2  # CJY for pos_masked

        # print("1")
        # 多尺度下的gcam进行融合
        overall_gcam = torch.cat(gcam_list, dim=1)
        # mean值法
        overall_gcam = torch.mean(overall_gcam, dim=1, keepdim=True)
        # max值法
        # overall_gcam = torch.max(overall_gcam, dim=1, keepdim=True)[0]

        # gcam_list = [overall_gcam]
        """
        #overall_gcam_index1 = torch.max(overall_gcam, dim=1, keepdim=True)[1]
        #overall_gcam = torch.max(overall_gcam, dim=1, keepdim=True)[0]
        overall_gcam_index = torch.max(overall_gcam.abs(), dim=1, keepdim=True)[1]
        overall_gcam_index_onehot = torch.nn.functional.one_hot(overall_gcam_index.permute(0, 2, 3, 1), target_layer_num).squeeze(3).permute(0, 3, 1, 2)
        #if overall_gcam_index_onehot.shape[1] == 1:
            #overall_gcam_index_onehot = overall_gcam_index_onehot + 1
        overall_gcam = overall_gcam * overall_gcam_index_onehot
        overall_gcam = torch.sum(overall_gcam, dim=1, keepdim=True)
        #overall_gcam = torch.relu(overall_gcam)  # 只保留正值
        #overall_gcam = torch.mean(overall_gcam, dim=1, keepdim=True)
        #overall_gcam = torch.relu(overall_gcam)
        gcam_list = [overall_gcam]
        #"""

        self.inter_output.clear()
        self.inter_gradient.clear()

        return gcam_list, gcam_max_list, overall_gcam

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
        gcam = gcam / (gcam_abs_max_expand.clamp(min=1E-12).detach())  # [-1,+1]
        # print("gcam_max{}".format(gcam_abs_max.mean().item()))
        return gcam, gcam_abs_max  # .mean().item()

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

    def set_hooks(self):
        # 0.different Network config
        Layer_Net_Dict={
            "MBagNet":["denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "DenseNet":["denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "ResNet": [],
            "Vgg":[],
        }

        # 1.GradCAM hook               GradCAM 如果其不为none，那么就设置hook
        self.target_layer = []
        if self.gcamState == True:
            # 用于存储中间的特征输出和对应的梯度
            self.inter_output = []
            self.inter_gradient = []

            self.target_layer = ["denseblock4"]  # ["denseblock3"]#["denseblock1", "denseblock2", "denseblock3", "denseblock4"]#, "denseblock2",
            if self.target_layer != []:
                for tl in self.target_layer:
                    for module_name, module in self.base.features.named_modules():
                        if module_name == tl:
                            print("Grad-CAM hook on ", module_name)
                            module.register_forward_hook(self.forward_hook_fn)
                            # module.register_backward_hook(self.backward_hook_fn)  不以backward求取gcam了，why，因为这种回传会在模型中保存梯度，然后再清零会出问题
                            break

        # 2.Guided Backpropagation Hook
        if self.guidedBP == True:
            print("Set GuidedBP Hook on Relu")
            for module_name, module in self.named_modules():
                if isinstance(module, torch.nn.ReLU) == True:
                    module.register_backward_hook(self.guided_backward_hook_fn)
        self.guidedBPstate = 0  # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上

        # 3.观测梯度 hook
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
                if i not in self.state_dict():
                    print("Cannot load %s, Maybe you are using incorrect framework"%i)
                    continue
                self.state_dict()[i].copy_(param_dict[i])

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