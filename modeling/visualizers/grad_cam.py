"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

#from .misc_functions import get_example_params, save_class_activation_images



class GradCAM():
    def __init__(self, model, num_classes, target_layer, useGuidedBP=False):
        #self.model = model
        self.num_classes = num_classes
        self.target_layer = target_layer  # 最好按forward顺序写
        self.num_terget_layer = len(self.target_layer)
        self.inter_output = []
        self.inter_gradient = []
        self.useGuidedBP = useGuidedBP
        self.GuidedBPstate = 0    # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上

        self.hookIndex = 0
        self.setHook(model)

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
                    print(module_name)
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
        if self.GuidedBPstate == True:
            pos_grad_out = grad_out[0].gt(0)
            result_grad = pos_grad_out * grad_in[0]
            return (result_grad,)
        else:
            pass

    # Obtain Gradient
    def ObtainGradient(self, logits, labels):
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


    # Generate Single CAM (backward)
    def GenerateCAM(self, inter_output, inter_gradient):
        # backward形式
        gcam = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
        gcam = gcam * (gcam.shape[-1] * gcam.shape[-2])  # 如此，形式上与最后一层计算的gcam量级就相同了  （由于最后loss使用mean，所以此处就不mean了）
        gcam = torch.relu(gcam)  # CJY at 2020.4.18
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
    def GenerateOverallCAM(self, gcam_list):
        # print("1")
        # 多尺度下的gcam进行融合
        overall_gcam = torch.cat(gcam_list, dim=1)
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
    def GenerateVisualiztions(self, logits, labels, visual_num):
        target_layer_num = len(self.target_layer)
        gcam_list = []
        gcam_max_list = [] # 记录每个Grad-CAM的归一化最大值

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
            gcam_list.append(norm_gcam)  # 将不同模块的gcam保存到gcam_list中
            gcam_max_list.append(gcam_max.detach().mean().item() / 2) # CJY for pos_masked

        # Generate Overall CAM
        overall_gcam = self.GenerateOverallCAM(gcam_list=gcam_list)

        # Clear Reservation
        #self.inter_output.clear()
        self.inter_gradient.clear()

        return gcam_list, gcam_max_list, overall_gcam



class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer, guided_back=False):
        self.model = model
        self.target_module_name = target_layer
        self.target_layer = 1
        self.gradients = None
        self.guided_back = guided_back  # 是否进行导向反向传播
        self.initialize()


    #def register_hooks(self):
    def set_requires_gradients(self, module, input, output):
        self.conv_out = output
        output.requires_grad_(True)
        output.register_hook(self.save_gradient)

    def set_requires_gradients_firstlayer(self, module, input, output):
        self.conv_out = input[0]
        input[0].requires_grad_(True)
        input[0].register_hook(self.save_gradient)

    def save_gradient(self, grad):
        self.gradients = grad

    # 第二种方法
    def backward_hook_fn(self, module, grad_in, grad_out):
        self.gradients = grad_in[1]
        #print(grad_in)
        #print(1)

    # Guided Backpropgation
    #用于Relu处的hook
    def guided_backward_hook_fn(self, module, grad_in, grad_out):
        #self.gradients = grad_in[1]
        pos_grad_out = grad_out[0].gt(0)
        result_grad = pos_grad_out * grad_in[0]
        return (result_grad,)


    def initialize(self):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        if self.guided_back == True:
            for module_name, module in self.model.named_modules():
                if isinstance(module, torch.nn.ReLU) == True:
                    module.register_backward_hook(self.guided_backward_hook_fn)

        for module_name, module in self.model.base.features.named_modules():  #此处的hook比较特殊，因为并非是寻找model的参数的梯度。而是要寻找输出特征的梯度。所以是与输入相关的，不能提前定义
            if 1:#isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.MaxPool2d) or isinstance(module, torch.nn.AvgPool2d) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.ReLU):
                #print(module_name)
                if self.target_module_name == "":
                    module.register_forward_hook(self.set_requires_gradients_firstlayer)
                    #module.register_backward_hook(self.backward_hook_fn)
                    break
                if module_name == self.target_module_name:
                    module.register_forward_hook(self.set_requires_gradients)
                    break


    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        final_logits = self.model(x)

        return self.conv_out, final_logits


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, guided_back=False, weight_fetch_type="Grad-CAM-pixelwise", show_pos=True):
        self.model = model
        self.model.eval()
        # Define extractor
        self.target_layer = target_layer

        self.weight_fetch_type = weight_fetch_type  # "Grad-CAM"， "Grad-CAM++"  "Grad-CAM-pixelwise"
        self.guided_back = guided_back

        self.extractor = CamExtractor(self.model, target_layer, self.guided_back)

        self.show_pos = show_pos


    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        input_image.requires_grad_(True)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output.cuda())#, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]


        # 1. Grad-CAM
        # Get weights from gradients
        if self.weight_fetch_type == "Grad-CAM":
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # 加权求取CAM
            # Create empty numpy array for cam
            cam = np.zeros(target.shape[1:], dtype=np.float32)  # 此处原本用one矩阵，但是由于梯度太小，导致增加量较少而被1稀释，所以改用0
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

        if self.weight_fetch_type == "Grad-PCAM":
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # 加权求取CAM
            # Create empty numpy array for cam
            cam = np.zeros(target.shape[1:], dtype=np.float32)  # 此处原本用one矩阵，但是由于梯度太小，导致增加量较少而被1稀释，所以改用0
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                w = np.maximum(w, 0)
                cam += w * target[i, :, :]

        # 2. Grad-CAM++   smooth function is exp(x)
        if self.weight_fetch_type == "Grad-CAM++":
            gradients = self.extractor.gradients.cpu()
            activations = conv_output.cpu()
            logits = model_output[:, target_class].cpu()
            b, k, u, v = gradients.shape
            alpha_num = gradients.pow(2)
            alpha_denom = gradients.pow(2).mul(2) + \
                          activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)

            alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))  #避免分母为0

            alpha = alpha_num.div(alpha_denom + 1e-7)
            positive_gradients = torch.relu(logits.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
            weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

            saliency_map = (weights * activations).sum(1, keepdim=True)
            cam = saliency_map.squeeze(0).squeeze(0).detach().numpy()
            #cam = torch.relu(cam)

        # 3. pixel-wise-Grad-CAM   直接对应element-wise 相乘
        if self.weight_fetch_type == "Grad-CAM-pixelwise":
            cam = np.sum(target * guided_gradients, axis=0)


        # 比较一下与logits的差距，毕竟是用线性逼近非线性
        #cam_sum = np.sum(cam, axis=(0,1))
        #logit = model_output[0][target_class]

        # 只取正的部分 我要是不取呢
        #cam = np.maximum(cam, 0)
        #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  * 0.5 +0.5# Normalize between 0-1

        #CJY 用abs来归一化
        #"""
        if self.show_pos == 1:
            cam = np.maximum(cam, 0)
        max = np.max(np.abs(cam))*2
        if max != 0:
            cam = cam / max + 0.5# Normalize between 0-1
        else:
            cam = cam + 0.5
        #"""

        """
        pcam = np.maximum(cam, 0)
        ncam = np.minimum(cam, 0)

        pmax = np.max(np.abs(pcam)) * 2
        nmax = np.max(np.abs(ncam)) * 2

        if pmax != 0 and nmax != 0:
            cam = pcam/pmax + ncam/nmax + 0.5# Normalize between 0-1
        elif pmax != 0 and nmax == 0:
            cam = pcam / pmax + 0.5  # Normalize between 0-1
        elif pmax == 0 and nmax != 0:
            cam = ncam/nmax + 0.5  # Normalize between 0-1
        else:
            cam = cam + 0.5

        #"""


        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]),Image.ANTIALIAS))/255   #  Image.ANTIALIAS
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
