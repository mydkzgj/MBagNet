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


from utils import featrueVisualization as fV


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
    def __init__(self,  base_name, num_classes,):
        super(Baseline, self).__init__()

        self.heatmapFlag = 0
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
        elif base_name == "multi_bagnet":
            self.base = mbagnet224(preAct=True, fusion="concat", reduction=1)
            self.heatmapFlag = 0
            if self.base.fusion == "concat":
                self.classifier_type = "receptive_field"
            self.in_planes = self.base.num_features

            self.num_receptive_field = self.base.num_receptive_field
            self.num_channel_per_rf = self.in_planes // self.num_receptive_field
            self.num_layers = self.base.num_layers
            self.receptive_field_list = self.base.receptive_field_list

        # 2.以下是classifier的网络结构
        if self.classifier_type == "normal":
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            self.classifier.apply(weights_init_classifier)
        elif self.classifier_type == "receptive_field":   # CJY at 2019.12.04  增加group分类   receptive field
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.rf_intra_classifier = nn.Conv2d(self.in_planes, self.num_receptive_field * self.num_classes, kernel_size=1,
                                                 stride=1, groups=self.num_receptive_field, bias=False)
            self.rf_inter_classifier = nn.Linear(self.num_receptive_field, 1)
            self.rf_intra_classifier.apply(weights_init_classifier)
            self.rf_inter_classifier.apply(weights_init_classifier)


        if self.heatmapFlag == 1:
            modules = self.named_modules()
            for name, module in modules:
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Identity):
                    if "reduction0" in name:
                        module.register_forward_hook(hook_fn_forward)
                    if "reduction1" in name:
                        module.register_forward_hook(hook_fn_forward)

        """
        self.feat_pos_weight = torch.tensor([[12544, 12544, 12544, 12544, 12544, 12544, 12544,
                                             3136, 3136, 3136, 3136, 3136, 3136, 3136, 3136,
                                             729, 729, 729, 729, 729, 729, 729, 729,]])#torch.arange(1,self.in_planes+1).float()/self.in_planes
        self.feat_pos_weight = self.feat_pos_weight.reshape(23,1).expand(23, 8)
        self.feat_pos_weight = self.feat_pos_weight.reshape(-1, 1).unsqueeze(-1)
        self.feat_pos_weight = self.feat_pos_weight.float().cuda()
        """
        print(self.base)

    #CJY 传入label
    def transmitLabel(self, label):
        self.label = label

    #CJY
    def generateScoreMap(self, show_maps, rf_feature_maps, rank_num_per_class=10):
        # CJY at 2019.12.5  计算不同感受野的图像的特征
        # 1.记录不同感受野的权值
        if self.classifier_type == "receptive_field":
            show_maps.append(self.rf_inter_classifier.weight.unsqueeze(-1).unsqueeze(-1))
        else:
            show_maps.append(torch.ones((1, self.num_receptive_field, 1, 1)))

        # 2.计算不同感受野下不同类别的预测热图  需要区分融合类型
        if self.classifier_type == "receptive_field":
            for rf_index in range(len(rf_feature_maps)):
                rf_intra_feature = rf_feature_maps[rf_index].unsqueeze(0)
                rf_intra_weight = self.rf_intra_classifier.weight[
                                  rf_index * self.num_classes:(rf_index + 1) * self.num_classes]
                rf_intra_logit = F.conv2d(rf_intra_feature, rf_intra_weight)
                rf_intra_logit = (rf_intra_logit * self.rf_inter_classifier.weight[0][rf_index])
                show_maps.append(rf_intra_logit)
        elif self.classifier_type == "normal":   #只有最终的分类器self.classifier
            for rf_index in range(len(rf_feature_maps)):
                rf_intra_feature = rf_feature_maps[rf_index].unsqueeze(0)
                rf_intra_weight = self.classifier.weight.unsqueeze(-1).unsqueeze(-1)
                rf_intra_logit = F.conv2d(rf_intra_feature, rf_intra_weight)
                show_maps.append(rf_intra_logit)

        # 3.找到logits最大的几个evidence
        #(1)将特征logits拉平连接在一起
        for i in range(3, len(show_maps)):
            rf_intra_logit = show_maps[i]
            if i == 3:
                rf_intra_logit_flatten = rf_intra_logit.view(rf_intra_logit.shape[0], rf_intra_logit.shape[1], -1)
            else:
                rf_intra_logit_flatten = torch.cat([rf_intra_logit_flatten, rf_intra_logit.view(rf_intra_logit.shape[0], rf_intra_logit.shape[1], -1)], dim=2)

            self.receptive_field_list[i-3]["width"] = rf_intra_logit.shape[-1]
            self.receptive_field_list[i-3]["size"] = rf_intra_logit.shape[-1]*rf_intra_logit.shape[-1]

        #(2)排序
        #rank_logits = torch.sort(rf_intra_logit_flatten, dim=2, descending=True)
        rank_logits_index = torch.argsort(rf_intra_logit_flatten, dim=2, descending=True)

        pick_logits_index = rank_logits_index[0][:, 0:rank_num_per_class]
        pick_logits = torch.ones_like(pick_logits_index).float()
        for i in range(self.num_classes):
            for j in range(rank_num_per_class):
                pick_logits[i][j] = rf_intra_logit_flatten[0][i][pick_logits_index[i][j]]

        #(3)依据index定位感受野层index和map中的（i，j）     #感受野大小，位置
        pick_logits_dict = {}
        for i in range(self.num_classes):
            pick_logits_dict[i] = []
            for j in range(rank_num_per_class):
                index = pick_logits_index[i][j].item()
                for rf_i in range(len(self.receptive_field_list)):   #遍历所有感受野信息
                    if index - self.receptive_field_list[rf_i]["size"] < 0:   #如果index<size，说明在在该感受野内，那么继续求取横纵坐标
                        h_index = index // self.receptive_field_list[rf_i]["width"]
                        w_index = index % self.receptive_field_list[rf_i]["width"]
                        padding = self.receptive_field_list[rf_i]["padding"]
                        rf_size = self.receptive_field_list[rf_i]["rf_size"]
                        rf_stride = self.receptive_field_list[rf_i]["rf_stride"]
                        center_x = -padding + rf_size//2 + 1 + w_index*rf_stride
                        center_y = -padding + rf_size//2 + 1 + h_index*rf_stride
                        break
                    else:
                        index -= self.receptive_field_list[rf_i]["size"]

                pick_logits_dict[i].append({"h":h_index, "w":w_index, "padding":padding, "rf_size":rf_size, "center_x":center_x, "center_y":center_y, "logit":pick_logits[i][j].item(),"max_padding":self.receptive_field_list[-1]["padding"]})

        # 可视化
        fV.showrfFeatureMap(show_maps, self.num_classes, pick_logits_dict)
        rf_feature_maps.clear()
        show_maps.clear()
        
    def forward(self, x):
        base_out = self.base(x)
        if self.classifier_type == "normal":
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            final_logits = self.classifier(feat)
        elif self.classifier_type == "receptive_field":
            #CJY at 2019.12.4
            global_feat = self.gap(base_out)  # (b, ?, 1, 1)
            rf_logits = self.rf_intra_classifier(global_feat)    #  (b, num_receptive_field*self.num_classes, 1, 1)
            rf_logits = rf_logits.view(rf_logits.shape[0], self.num_receptive_field, -1).permute(0, 2, 1)  #  (b, num_receptive_field, self.num_classes)
            #rf_score = torch.softmax(rf_logits, dim=1)
            weighted_rf_logits = rf_logits * self.rf_inter_classifier.weight
            final_logits = self.rf_inter_classifier(rf_logits).squeeze(-1)

        # 可视化
        if self.heatmapFlag == 1:
            global show_maps, rf_feature_maps
            sample_index = 0
            # predict_label
            self.p_label = torch.argmax(final_logits, dim=1)
            show_maps.insert(0, x[sample_index])
            show_maps.insert(1, [self.label[sample_index], self.p_label[sample_index]])
            #a = weighted_rf_logits[0].cpu().detach().numpy()
            #a = final_logits[0].cpu().detach().numpy()
            self.generateScoreMap(show_maps, rf_feature_maps, 10)

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

        return global_feat, final_logits


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
