"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

#from .misc_functions import get_example_params, save_class_activation_images

from .draw_tool import draw_visualization


class GuidedBackpropagation():
    def __init__(self, model, num_classes,):
        self.model = model
        self.num_classes = num_classes
        self.target_layer = [""] #注：需要让input可计算梯度； target_layer  # 最好按forward顺序写
        self.num_terget_layer = len(self.target_layer)
        self.inter_output = []
        self.inter_gradient = []
        self.useGuidedBP = True
        self.guidedBPstate = 0    # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上

        self.hookIndex = 0
        self.setHook(model)

        self.draw_index = 0

    def setHook(self, model):
        print("Set Hook for Visualization:")
        # 0.different Network config
        Layer_Net_Dict = {
            "MBagNet": ["denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "DenseNet": ["denseblock1", "denseblock2", "denseblock3", "denseblock4"],
            "ResNet": [],
            "Vgg": [],
        }

        # 1.Set Feature-Reservation Hook
        # 用于存储中间的特征输出和对应的梯度
        if self.target_layer != []:
            for tl in self.target_layer:
                for module_name, module in model.named_modules():
                    #print(module_name)
                    if tl == "": # 对输入图像进行hook
                        module.register_forward_hook(self.set_requires_gradients_firstlayer)
                        break
                    if module_name == tl:
                        print("Visualization Hook on ", module_name)
                        module.register_forward_hook(self.reserve_features_hook_fn)
                        #module.register_backward_hook(self.backward_hook_fn)  不以backward求取gcam了，why，因为这种回传会在模型中保存梯度，然后再清零会出问题
                        break
        else:
            raise Exception("Without target layer can not generate Visualization")

        # 2.Set Guided-Backpropagation Hook
        if self.useGuidedBP == True:
            print("Set GuidedBP Hook on Relu")
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU) == True:
                    module.register_backward_hook(self.guided_backward_hook_fn)

    # Hook Function
    def set_requires_gradients_firstlayer(self, module, input, output):
        # 为了避免多次forward，保存多个特征，所以通过计数完成置零操作
        self.inter_output.clear()
        self.inter_gradient.clear()
        #input[0].requires_grad_(True)   # 在这里改input的grad好像没用；只能在forward之前更改
        self.inter_output.append(input[0])

    def reserve_features_hook_fn(self, module, input, output):
        # 为了避免多次forward，保存多个特征，所以通过计数完成置零操作
        if self.hookIndex % self.num_terget_layer == 0:
            self.hookIndex = 0
            self.inter_output.clear()
            self.inter_gradient.clear()
        self.inter_output.append(output)
        self.hookIndex = self.hookIndex + 1

    def guided_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedBPstate == True:
            pos_grad_out = grad_out[0].gt(0)
            result_grad = pos_grad_out * grad_in[0]
            return (result_grad,)
        else:
            pass

    # Obtain Gradient
    def ObtainGradient(self, logits, labels):
        self.observation_class = labels.cpu().numpy().tolist()
        # 将label转为one - hot
        gcam_one_hot_labels = torch.nn.functional.one_hot(labels, self.num_classes).float()
        #gcam_one_hot_labels = gcam_one_hot_labels.to(device) if torch.cuda.device_count() >= 1 else gcam_one_hot_labels
        try:
            labels.get_device()
            gcam_one_hot_labels = gcam_one_hot_labels.cuda()
        except:
            pass

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
        gcam = gcam / gcam_abs_max_expand.clamp(min=1E-12).detach() *0.5 + 0.5  # [-1, 1]
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


    # Generate Single CAM (backward)
    def GenerateCAM(self, inter_output, inter_gradient):
        # backward形式
        gcam = inter_gradient
        #gcam = torch.sum(gcam.abs(), dim=1, keepdim=True)
        return gcam

    """
    def GenerateCAM(self):
        # forward  # 最后一层是denseblock4的输出，使用forward形式        
        gcam = F.conv2d(inter_output, model.classifier.weight.unsqueeze(-1).unsqueeze(-1))
        # gcam = gcam /(gcam.shape[-1]*gcam.shape[-2])  #如此，形式上与其他层计算的gcam量级就相同了
        # gcam = torch.softmax(gcam, dim=-1)
        pick_label = labels[labels.shape[0] - gcamBatchDistribution[1]:labels.shape[0]]
        pick_list = []
        for j in range(pick_label.shape[0]):
            pick_list.append(gcam[j, pick_label[j]].unsqueeze(0).unsqueeze(0))
        gcam = torch.cat(pick_list, dim=0)
    #"""


    # Generate Overall CAM
    def GenerateOverallCAM(self, gcam_list, input_size):
        # print("1")
        # 多尺度下的gcam进行融合
        resized_gcam_list = []
        for gcam in gcam_list:
            gcam = torch.nn.functional.interpolate(gcam, input_size, mode='bilinear')  # mode='nearest'  'bilinear'
            resized_gcam_list.append(gcam)
        overall_gcam = torch.cat(resized_gcam_list, dim=1)
        # mean值法
        overall_gcam = torch.mean(overall_gcam, dim=1, keepdim=True)
        # max值法
        # overall_gcam = torch.max(overall_gcam, dim=1, keepdim=True)[0]

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
        return overall_gcam


    # Generate Visualiztions Function   # 统一用GenerateVisualiztion这个名字吧
    def GenerateVisualiztions(self, logits, labels, input_size, visual_num):
        target_layer_num = len(self.target_layer)
        self.gcam_list = []
        self.gcam_max_list = [] # 记录每个Grad-CAM的归一化最大值

        # obtain gradients
        self.ObtainGradient(logits, labels)

        for i in range(target_layer_num):
            # 1.获取倒数visual_num个样本的activation以及gradient
            batch_num = logits.shape[0]
            visual_num = visual_num #gcamBatchDistribution[1]
            inter_output = self.inter_output[i][batch_num - visual_num:batch_num]  # 此处分离节点，别人皆不分离  .detach()
            inter_gradient = self.inter_gradient[i][batch_num - visual_num:batch_num]

            # 2.生成CAM
            gcam = self.GenerateCAM(inter_output, inter_gradient)

            # 3.Post Process
            # Amplitude Normalization
            norm_gcam, gcam_max = self.gcamNormalization(gcam)
            # Resize Interpolation
            # gcam = torch.nn.functional.interpolate(gcam, (seg_gt_masks.shape[-2], seg_gt_masks.shape[-1]), mode='bilinear')  #mode='nearest'  'bilinear'

            # 4.Save in List
            self.gcam_list.append(norm_gcam)  # 将不同模块的gcam保存到gcam_list中
            self.gcam_max_list.append(gcam_max.detach().mean().item() / 2) # CJY for pos_masked

        # Generate Overall CAM
        self.overall_gcam = self.GenerateOverallCAM(gcam_list=self.gcam_list, input_size=input_size)

        # Clear Reservation
        #self.inter_output.clear()
        self.inter_gradient.clear()

        return self.gcam_list, self.gcam_max_list, self.overall_gcam

    def DrawVisualization(self, imgs, labels, plabels, gtmasks, threshold, savePath):
        """
        :param imgs: 待可视化图像
        :param labels: 对应的label
        :param plabels: 预测的label
        :param gtmasks: 掩膜真值
        :param threshold: 二值化阈值 0-1
        :param savePath: 存储路径
        :return:
        """
        for j in range(imgs.shape[0]):
            for i, gcam in enumerate(self.gcam_list):
                layer_name = self.target_layer[i]
                label_prefix = "L{}_P{}".format(labels[j].item(), plabels[j].item())
                visual_prefix = layer_name.replace(".", "-") + "_S{}".format(self.observation_class[j])
                draw_visualization(imgs[j], gcam[j], gtmasks[j], threshold, savePath, str(self.draw_index), label_prefix, visual_prefix)
            self.draw_index = self.draw_index + 1
        return 0






