"""
Created on 2020.7.4

@author: Jiayang Chen - github.com/mydkzgj
"""

from .grad_cam import *
from .guided_backpropagation import *

from .draw_tool import draw_visualization


class GuidedGradCAM():
    def __init__(self, model, num_classes, target_layer, useGuidedBP=False):
        self.num_classes = num_classes
        self.target_layer = target_layer  #注：需要让input可计算梯度； target_layer  # 最好按forward顺序写
        self.draw_index = 0

        # 子模块
        self.gcam = GradCAM(model, num_classes, target_layer=[target_layer[-2]], useGuidedBP=False)
        self.guidedBP = GuidedBackpropagation(model, num_classes)

        self.gcam.reservePos = True
        self.reservePos = True#self.guidedBP.reservePos

        self.process_type = "max" #"reserve", "max", "sum"


    # Generate Visualiztions Function   # 统一用GenerateVisualiztion这个名字吧
    def GenerateVisualiztions(self, logits, labels, input_size, visual_num):
        self.observation_class = labels.cpu().numpy().tolist()
        gcam_list, self.gcam_max_list, self.overall_gcam = self.gcam.GenerateVisualiztions(logits, labels, input_size, visual_num)
        gbp_list, _, _ = self.guidedBP.GenerateVisualiztions(logits, labels, input_size, visual_num)

        gbp = gbp_list[0]  #注：gbp为3通道，以0.5为baseline
        self.gcam_list = []
        for i, cam in enumerate(gcam_list):
            resized_cam = torch.nn.functional.interpolate(cam, input_size, mode='bilinear')
            if self.process_type == "reserve":
                # 版本0：保留三通道
                gbp = gbp - 0.5
            elif self.process_type == "sum":
                # 版本1：求和，形成单通道
                gbp = (gbp - 0.5).abs().sum(dim=1, keepdims=True)
            elif self.process_type == "max":
                gbp = torch.max((gbp - 0.5).abs(), dim=1, keepdim=True)[0]
            else:
                raise Exception("Wrong process type!")
            ggcam = resized_cam * gbp
            norm_ggcam, ggcam_max = self.guidedBP.gcamNormalization(ggcam)  #使用guidedBP的归一化，即reserve-pos为False
            self.gcam_list.append(norm_ggcam)

        return self.gcam_list, self.gcam_max_list, self.overall_gcam

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
            "binary_visualization_on_image": 1,
            "color_visualization_on_image": 0,
            "binary_visualization_on_segmentation": 0,
            "color_visualization_on_segmentation": 0,
            "segmentation_ground_truth": 1,
        }

        for j in range(imgs.shape[0]):
            for i, gcam in enumerate(self.gcam_list):
                layer_name = self.target_layer[i]
                label_prefix = "L{}_P{}".format(labels[j].item(), plabels[j].item())
                visual_prefix = layer_name.replace(".", "-") + "_S{}".format(self.observation_class[j])
                if gtmasks is not None:
                    draw_visualization(imgs[j], gcam[j], gtmasks[j], threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)
                else:
                    draw_visualization(imgs[j], gcam[j], None, threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)
        return 0






