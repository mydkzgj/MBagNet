"""
Created on 2020.7.4

@author: Jiayang Chen - github.com/mydkzgj
"""

from .cjy_dual_backpropagation import *
from .cjy_contrastive_guided_pgrad_cam import *

from .draw_tool import draw_visualization


class CJY_CONTRASTIVE_GUIDED_PGRAD_CAM_MULTIPLY_DUAL_BACKPROPAGATION():
    def __init__(self, model, num_classes, target_layer, guided_type="grad"):
        self.num_classes = num_classes
        self.target_layer = target_layer
        self.draw_index = 0

        self.guided_type = guided_type

        # 子模块

        self.db_pgcam = CJY_DUAL_BACKPROPAGATION(model, num_classes, target_layer=self.target_layer)
        self.cg_pgcam = CJY_CONTRASTIVE_GUIDED_PGRAD_CAM(model, num_classes, target_layer=self.target_layer, guided_type=self.guided_type)

        if self.db_pgcam.multiply_input == self.cg_pgcam.multiply_input:
            self.multiply_input = self.cg_pgcam.multiply_input
        else:
            raise Exception("Num Multiply Input Not Match!")

        self.reservePos = True

        self.db_pgcam.reservePos = False
        self.cg_pgcam.reservePos = True

        self.threshold = 0.1

    # Generate Visualiztions Function   # 统一用GenerateVisualiztion这个名字吧
    def GenerateVisualiztions(self, logits, labels, input_size, visual_num):
        self.observation_class = labels.cpu().numpy().tolist()
        db_pgcam_list, _, _ = self.db_pgcam.GenerateVisualiztions(logits, labels, input_size, visual_num)
        cg_pgcam_list, _, _ = self.cg_pgcam.GenerateVisualiztions(logits, labels, input_size, visual_num)

        self.gcam_list = []
        self.gcam_max_list = []
        for i in range(len(cg_pgcam_list)):
            db_pgcam = db_pgcam_list[i]
            cg_pgcam = cg_pgcam_list[i]

            cam = 2 * (db_pgcam - 0.5) * cg_pgcam.gt(self.threshold).float()

            norm_cam, cam_max = self.gcamNormalization(cam)  #使用guidedBP的归一化，即reserve-pos为False
            self.gcam_list.append(norm_cam)
            self.gcam_max_list.append(cam_max)

        self.overall_gcam = None

        return self.gcam_list, self.gcam_max_list, self.overall_gcam

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
        for i in range(gcam.ndimension()-1):
            gcam_abs_max = gcam_abs_max.unsqueeze(-1)
        gcam_abs_max_expand = gcam_abs_max.clamp(1E-12).expand_as(gcam)
        gcam = gcam / (gcam_abs_max_expand.clamp(min=1E-12).detach())  # [-1,+1]
        if self.reservePos != True:
            gcam = gcam * 0.5 + 0.5                                    # [0, 1]
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

    def DrawVisualization(self, imgs, labels, plabels, gtmasks, threshold, savePath, imgsName):
        """
        :param imgs: 待可视化图像
        :param labels: 对应的label
        :param plabels: 预测的label
        :param gtmasks: 掩膜真值
        :param threshold: 二值化阈值 0-1
        :param savePath: 存储路径
        :return:
        """
        draw_flag_dict = {
            "originnal_image": 1,
            "gray_visualization": 0,
            "binary_visualization": 0,
            "color_visualization": 1,
            "binary_visualization_on_image": 0,
            "color_visualization_on_image": 0,
            "binary_visualization_on_segmentation": 0,
            "color_visualization_on_segmentation": 0,
            "segmentation_ground_truth": 1,
        }

        for j in range(imgs.shape[0]):
            labels_str = ""
            plabels_str = ""
            if len(labels.shape) == 1:
                labels_str = str(labels[j].item())
                plabels_str = str(plabels[j].item())
            elif len(labels.shape) == 2:
                for k in range(labels.shape[1]):
                    labels_str = labels_str + "-" + str(labels[j][k].item())
                for k in range(plabels.shape[1]):
                    plabels_str = plabels_str + "-" + str(plabels[j][k].item())
                labels_str = labels_str.strip("-")
                plabels_str = plabels_str.strip("-")
            label_prefix = "L{}_P{}".format(labels_str, plabels_str)
            # label_prefix = "L{}_P{}".format(labels[j].item(), plabels[j].item())

            for i, gcam in enumerate(self.gcam_list):
                layer_name = self.target_layer[i]
                visual_prefix = layer_name.replace(".", "-") + "_S{}".format(self.observation_class[j])
                if gtmasks is not None:
                    draw_visualization(imgs[j], gcam[j], gtmasks[j], threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)
                else:
                    draw_visualization(imgs[j], gcam[j], None, threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)
        return 0





