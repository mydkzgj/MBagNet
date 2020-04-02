"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

from .misc_functions import get_example_params, save_class_activation_images





class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_module_name = target_layer
        self.target_layer = 1
        self.gradients = None
        self.guided_back = True  # 是否进行导向反向传播
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
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.target_layer = target_layer
        self.extractor = CamExtractor(self.model, target_layer)

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
        weight_fetch_type = "Grad-CAM-pixelwise"  #"Grad-CAM++"  "Grad-CAM-pixelwise"

        # 1. Grad-CAM
        # Get weights from gradients
        if weight_fetch_type == "Grad-CAM":
            weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
            # 加权求取CAM
            # Create empty numpy array for cam
            cam = np.zeros(target.shape[1:], dtype=np.float32)  # 此处原本用one矩阵，但是由于梯度太小，导致增加量较少而被1稀释，所以改用0
            # Multiply each weight with its conv output and then, sum
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]

        # 2. Grad-CAM++   smooth function is exp(x)
        if weight_fetch_type == "Grad-CAM++":
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
        if weight_fetch_type == "Grad-CAM-pixelwise":
            cam = np.sum(target * guided_gradients, axis=0)


        # 比较一下与logits的差距，毕竟是用线性逼近非线性
        #cam_sum = np.sum(cam, axis=(0,1))
        #logit = model_output[0][target_class]

        # 只取正的部分 我要是不取呢
        #cam = np.maximum(cam, 0)
        #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  * 0.5 +0.5# Normalize between 0-1

        #CJY 用abs来归一化
        #"""
        #cam = np.maximum(cam, 0)
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
                       input_image.shape[3]),))/255   #  Image.ANTIALIAS
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
