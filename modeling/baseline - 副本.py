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




from ptflops import get_model_complexity_info


rf_feature_maps = []
show_maps = []

def hook_fn_forward(module, input, output):
    #为了减小显存占用，我们一次只保存一维
    global rf_feature_maps
    rf_feature_maps.append(output[0])



'''
similarity_maps = []
def cosine_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    feat_num = x.size(1)
    m, n = x.size(0), y.size(0)

    xx = x.expand(n, m, feat_num)
    yy = y.expand(m, n, feat_num).transpose(1, 0)

    dist = F.cosine_similarity(xx, yy, dim=2)
    return dist


# 计算C维特征的相关性
def features_simlarity(features):
    return cosine_dist_rank(features.view(features.shape[0], -1), features.view(features.shape[0], -1))


# 定义 forward hook function
def hook_fn_forward(module, input, output):
    #print(module)  # 用于区分模块
    #fV.maps_show3((output+0.5).clamp(0, 1), output.shape[1])
    #fV.maps_show3(input[0], input[0].shape[1])
    similarity_map = features_simlarity(output)
    similarity_maps.append(similarity_map.unsqueeze(dim=0))  # 然后分别存入全局 list 中
'''


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

        self.heatmapFlag = 0
        self.hookFlag = 0
        self.num_classes = num_classes
        self.base_name = base_name
        self.classifier_type = "normal"   #默认是一层线性分类器

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
            self.base = mbagnet121(preAct=preAct, fusionType=fusionType, reduction=1, num_classes=self.num_classes)
            self.heatmapFlag = 1
            self.hookFlag = 0
            self.classifier_type = "normal"
            self.in_planes = self.base.num_features

        elif base_name == "multi_bagnet":
            self.base = mbagnetS224(preAct=preAct, fusionType=fusionType, reduction=1, num_classes=self.num_classes)
            self.heatmapFlag = 0
            self.hookFlag = 0
            if self.base.fusionType == "concat":
                self.classifier_type = "normal"#"receptive_field"
            self.in_planes = self.base.num_features

            self.num_receptive_field = self.base.num_receptive_field
            self.num_channel_per_rf = self.in_planes // self.num_receptive_field
            #self.num_layers = self.base.num_layers
            self.receptive_field_list = self.base.receptive_field_list
            self.rf_size_list = [self.receptive_field_list[i]["rf_size"] for i in range(len(self.receptive_field_list))]

        # 2.以下是classifier的网络结构
        if self.classifier_type == "normal":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)
        elif self.classifier_type == "receptive_field":   # CJY at 2019.12.04  增加group分类   receptive field
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.rf_intra_classifier = nn.Conv2d(self.in_planes, self.num_receptive_field * self.num_classes, kernel_size=1, # groups=self.num_receptive_field,
                                                 stride=1,  bias=False)
            self.rf_inter_classifier = nn.Linear(self.num_receptive_field, 1)
            self.rf_intra_classifier.apply(weights_init_classifier)
            self.rf_inter_classifier.apply(weights_init_classifier)

            #CJY 依据感受野大小对logits进行惩罚
            self.rf_size_weight = torch.tensor(self.rf_size_list).float()
            self.rf_size_weight = nn.Parameter(self.rf_size_weight/torch.sum(self.rf_size_weight), requires_grad=False)

        if self.hookFlag == 1:
            if self.base_name == "multi_bagnet":
                self.base.setReductionHook(hook_fn_forward)

            elif self.base_name == "densenetS224":
                modules = self.named_modules()
                for name, module in modules:
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.AvgPool2d) or isinstance(module, nn.MaxPool2d):
                        if "pool0" in name:
                            module.register_forward_hook(hook_fn_forward)
                            print(name)
                        if "conv2" in name:
                            module.register_forward_hook(hook_fn_forward)
                            print(name)
                        if "transition" in name and "pool" in name:
                            module.register_forward_hook(hook_fn_forward)
                            print(name)


        #print(self.base)
        #print(self.count_param())
        #print(self.count_param2())

    def count_param2(model):
        with torch.cuda.device(0):
            flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            return ('{:<30}  {:<8}'.format('Computational complexity: ', flops)) + ('{:<30}  {:<8}'.format('Number of parameters: ', params))

    def count_param(model):
        param_count = 0
        for param in model.parameters():
            param_count += param.view(-1).size()[0]
        return param_count

    #CJY 传入label
    def transmitLabel(self, label):
        self.label = label


    def forward(self, x):
        global show_maps, rf_feature_maps

        if self.base.classifierType == "none":   #当base只提供特征时
            base_out = self.base(x)
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            final_logits = self.classifier(feat)

        else:  #mbagnet专属
            final_logits, rf_logits_reserve = self.base(x)

            if self.heatmapFlag == 1:
                sample_index = 0  #选择显示的样本的索引
                self.p_label = torch.argmax(final_logits, dim=1)  # predict_label
                #1.存入样本
                show_maps.insert(0, x[sample_index])
                #2.存入标签
                show_maps.insert(1, [self.label[sample_index], self.p_label[sample_index]])
                #3.存入感受野的weight
                if self.base.classifierType == "receptive_field":
                    show_maps.append(self.base.rf_inter_classifier.weight.unsqueeze(-1).unsqueeze(-1))
                else:
                    show_maps.append(torch.ones((1, len(rf_logits_reserve), 1, 1)))
                #4.存入rf_logits   n个感受野+1个总和
                for i in range(len(rf_logits_reserve)):
                    show_maps.append(rf_logits_reserve[i][sample_index].unsqueeze(0))

                #5. show_map中成分如下：0.img 1.[label,predict_label] 2.n+1 个rf_logits
                self.base.generateScoreMap2(show_maps, num_classes=6, rank_num_per_class=10)

        return final_logits, final_logits, final_logits




        """
        #elif self.classifier_type == "receptive_field":
            #CJY at 2019.12.4
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            rf_logits = self.rf_intra_classifier(global_feat)    #  (b, num_receptive_field*self.num_classes, 1, 1)
            rf_logits = rf_logits.view(rf_logits.shape[0], self.num_receptive_field, -1).permute(0, 2, 1)  #  (b, num_receptive_field, self.num_classes)
            #rf_score = torch.softmax(rf_logits, dim=1)
            weighted_rf_logits = rf_logits * self.rf_inter_classifier.weight
            final_logits = self.rf_inter_classifier(rf_logits).squeeze(-1)


            #选取其中一个感受野进行logits的判断
            #final_logits = weighted_rf_logits[:,:,-3]


            # CJY 依据感受野大小对logits进行惩罚
            r1 = torch.sum(weighted_rf_logits.abs(), dim=1)
            r1 = r1/torch.sum(r1,dim=1, keepdim=True)
            rf_loss = torch.matmul(r1, self.rf_size_weight).mean()
        

            
            #CJY logits 稀疏化  logits abs 之和  其实就是norm1

            rf_intra_norm1_list = []
            rf_inter_norm1 = 0
            for rf_index in range(len(rf_feature_maps)):
                rf_intra_feature = rf_feature_maps[rf_index].unsqueeze(0)
                rf_intra_weight = self.rf_intra_classifier.weight[
                                  rf_index * self.num_classes:(rf_index + 1) * self.num_classes]
                rf_intra_logit = F.conv2d(rf_intra_feature, rf_intra_weight)
                rf_intra_logit = (rf_intra_logit * self.rf_inter_classifier.weight[0][rf_index])
                rf_intra_norm1 = rf_intra_logit.abs().view(rf_intra_logit.shape[0], rf_intra_logit.shape[1], -1).mean(dim=2, keepdim=True)
                if rf_index == 0:
                    rf_intra_norm1_map = rf_intra_norm1
                else:
                    rf_intra_norm1_map = torch.cat([rf_intra_norm1_map, rf_intra_norm1], dim=2)

                rf_inter_norm1 = rf_intra_norm1_map.mean()
                #a = rf_intra_norm1_map.squeeze(0).cpu().detach().numpy()
            
            rf_loss = rf_inter_norm1
            if self.heatmapFlag == 0:
                rf_feature_maps.clear()
            

        """
        # 可视化
        """
        if self.heatmapFlag == 1:
            sample_index = 0
            # predict_label
            self.p_label = torch.argmax(final_logits, dim=1)
            show_maps.insert(0, x[sample_index])
            show_maps.insert(1, [self.label[sample_index], self.p_label[sample_index]])
            #a = weighted_rf_logits[0].cpu().detach().numpy()
            #a = final_logits[0].cpu().detach().numpy()
            if self.classifier_type == "normal":
                # 计算一下全局的按类别特征图
                final_class_predict_map = F.conv2d(base_out, self.classifier.weight)
                rf_feature_maps.append(final_class_predict_map[0])

                self.base.generateScoreMap(show_maps,
                                           rf_feature_maps,
                                           base_out,
                                           num_classes=self.num_classes,
                                           classifier_type=self.classifier_type,
                                           classifier=self.classifier,
                                           rank_num_per_class=10)
            elif self.classifier_type == "receptive_field":
                # 计算一下全局的按类别特征图
                final_class_predict_logits = self.rf_intra_classifier(base_out)
                final_class_predict_logits = final_class_predict_logits.view(final_class_predict_logits.shape[0],
                                                                             -1, self.num_classes,
                                                                             final_class_predict_logits.shape[2],
                                                                             final_class_predict_logits.shape[3]).permute(0, 2, 3, 4, 1)
                final_class_predict_map = self.rf_inter_classifier(final_class_predict_logits).squeeze(-1)
                rf_feature_maps.append(final_class_predict_map[0])
                self.base.generateScoreMap(show_maps,
                                           rf_feature_maps,
                                           base_out,
                                           num_classes=self.num_classes,
                                           classifier_type=self.classifier_type,
                                           rf_intra_classifier=self.rf_intra_classifier,
                                           rf_inter_classifier=self.rf_inter_classifier,
                                           rank_num_per_class=10)

        #"""

        """
        # 计算每组特征的预测结果与标签的交叉熵
        rfs_l = rf_l * self.rfs_classifier.weight
        rfs_l2 = rf_l * (self.rfs_classifier.weight/torch.abs(self.rfs_classifier.weight))
        rfs_l3 = rf_l * (-1)
        target = self.label.unsqueeze(1).expand((rf_score.shape[0],rf_score.shape[2]))
        every_rf_loss_before_c = torch.mean(F.cross_entropy(rf_l, target, reduction='none'),dim=0)
        every_rf_loss_after_c = torch.mean(F.cross_entropy(rfs_l, target, reduction='none'),dim=0)
        every_rf_loss_after_c2 = torch.mean(F.cross_entropy(rfs_l2, target, reduction='none'), dim=0)
        every_rf_loss_after_c3 = torch.mean(F.cross_entropy(rfs_l3, target, reduction='none'), dim=0)
        loss = F.cross_entropy(score, self.label, reduction='none')

        #以下是sigmoid
        #one_hot_labels = torch.nn.functional.one_hot(self.label, score.shape[1]).float()
        #one_hot_labels = one_hot_labels.cuda() if torch.cuda.device_count() >= 1 else one_hot_labels
        #rf_l_sig = torch.sigmoid(rf_l).permute(0,2,1)
        #rfs_l_sig = torch.sigmoid(rfs_l).permute(0,2,1)
        #rfs_l2_sig = torch.sigmoid(rfs_l2).permute(0,2,1)
        #rfs_l3_sig = torch.sigmoid(rfs_l3).permute(0,2,1)
        #one_hot_labels = one_hot_labels.unsqueeze(1).expand_as(rfs_l_sig)
        #every_rf_loss_before_c_sig = torch.mean(torch.mean(F.binary_cross_entropy(rf_l_sig, one_hot_labels, reduction="none"),dim=-1),dim=0)
        #every_rf_loss_after_c_sig = torch.mean(torch.mean(F.binary_cross_entropy(rfs_l_sig, one_hot_labels, reduction="none"),dim=-1),dim=0)
        #every_rf_loss_after_c2_sig = torch.mean(torch.mean(F.binary_cross_entropy(rfs_l2_sig, one_hot_labels, reduction="none"), dim=-1), dim=0)
        #every_rf_loss_after_c3_sig = torch.mean(torch.mean(F.binary_cross_entropy(rfs_l3_sig, one_hot_labels, reduction="none"), dim=-1), dim=0)
        #"""

        #return global_feat, final_logits, global_feat#rf_loss


    def load_param(self, loadChoice, model_path):
        param_dict = torch.load(model_path)

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
                if i not in self.base.state_dict():
                    print(i)
                    continue
                self.base.state_dict()[i].copy_(param_dict[i])

        elif loadChoice == "Overall":
            for i in param_dict:
                if i not in self.state_dict():
                    print(i)
                    continue
                self.state_dict()[i].copy_(param_dict[i])

        elif loadChoice == "Classifier":
            for i in param_dict:
                if i not in self.classifier.state_dict():
                    continue
                self.classifier.state_dict()[i].copy_(param_dict[i])

"""
    # CJY
    def generateScoreMap(self, rf_score_maps, rf_feature_maps):
        # CJY at 2019.12.5  计算不同感受野的图像的特征
        img_width = rf_score_maps[0].shape[-1]
        # 1.计算不同感受野的权值
        rfs_score = self.rf_inter_classifier.weight
        rf_score_maps.append(rfs_score.unsqueeze(-1).unsqueeze(-1))

        # 2.计算不同感受野下不同类别的预测热图
        denlayer_num = self.num_layers.copy()
        denlayer_num[0] = denlayer_num[0] + 1  # [6 + 1, 12, 24, 16]

        transition_weight_list = [self.base.features.transition1.conv.weight,
                                  self.base.features.transition2.conv.weight,
                                  ]  # self.base.features.transition3.conv.weight]
        for rf_index in range(len(rf_feature_maps)):
            rf_feature = rf_feature_maps[rf_index]
            rf_weight = self.rf_intra_classifier.weight[rf_index * self.num_classes:(rf_index + 1) * self.num_classes]

            # 寻找该特征所在的block和layer的index
            ri = rf_index
            for index, dn in enumerate(denlayer_num):
                if dn > ri:
                    block_index = index
                    layer_index = ri
                    # print(block_index, layer_index, index)
                    break
                else:
                    ri = ri - dn

            # 找到其经过的的transition层weight
            rf_feat_tran = rf_feature.unsqueeze(0)
            for i in range(len(denlayer_num) - block_index - 1):
                transition_index = block_index + i
                shape = transition_weight_list[transition_index].shape
                in_channel = shape[1]
                out_channel = in_channel // 2
                t_weight = transition_weight_list[transition_index][rf_index * out_channel:(rf_index + 1) * out_channel]
                rf_feat_tran = F.conv2d(rf_feat_tran, t_weight)

            rf_logit_c = F.conv2d(rf_feat_tran, rf_weight)

            # CJY 如果是线性分类器
            rf_logit_c = rf_logit_c * self.rf_inter_classifier.weight[0][
                rf_index]  # /(rf_logit_c.shape[2]*rf_logit_c.shape[3])

            rf_score_map = rf_logit_c.view(rf_logit_c.shape[0], rf_logit_c.shape[1],
                                           -1)  # rf_logit_c.view(rf_logit_c.shape[0], rf_logit_c.shape[1], -1)  # F.softmax(rf_logit_c.view(rf_logit_c.shape[0],rf_logit_c.shape[1],-1), dim=2)
            rf_score_maps.append(rf_score_map.view_as(rf_logit_c))

        # CJY 找到logits最大的几个evidence
        rank_num = 10
        # 拉平连接
        for i in range(3, len(rf_score_maps)):
            rf_score_map = rf_score_maps[i]
            if i == 3:
                rf_score_flatten = rf_score_map.view(rf_score_map.shape[0], rf_score_map.shape[1], -1)
            else:
                rf_score_flatten = torch.cat(
                    [rf_score_flatten, rf_score_map.view(rf_score_map.shape[0], rf_score_map.shape[1], -1)], dim=2)
            width = rf_score_map.shape[-1]
            size = rf_score_flatten.shape[-1]
            self.receptive_field_list[i - 3]["width"] = width
            self.receptive_field_list[i - 3]["size"] = width * width

        # 排序
        sort_logits = torch.sort(rf_score_flatten, dim=2, descending=True)
        logits_index = torch.argsort(rf_score_flatten, dim=2, descending=True)

        rank_logits_index = logits_index[0][:, 0:rank_num]
        rank_logits = torch.ones_like(rank_logits_index).float()
        for i in range(self.num_classes):
            for j in range(rank_num):
                rank_logits[i][j] = rf_score_flatten[0][i][rank_logits_index[i][j]]

        # 依据index定位感受野层index和map中的（i，j）     #感受野大小，位置
        rank_logits_dict = {}
        for i in range(self.num_classes):
            rank_logits_dict[i] = []
            for j in range(rank_num):
                index = rank_logits_index[i][j].item()
                for rf_i in range(len(self.receptive_field_list)):
                    if index - self.receptive_field_list[rf_i]["size"] < 0:  # 说明在该感受野内，那么继续求取横纵坐标
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

                rank_logits_dict[i].append(
                    {"h": h_index, "w": w_index, "padding": padding, "rf_size": rf_size, "center_x": center_x,
                     "center_y": center_y, "max_padding": self.receptive_field_list[-1]["padding"]})

        # 可视化
        fV.showrfFeatureMap(rf_score_maps, self.num_classes, rank_logits_dict)
        rf_feature_maps = []
        rf_score_maps = []
"""
