"""
Created on 2020.7.4

@author: Jiayang Chen - github.com/mydkzgj
"""

import torch
from .draw_tool import draw_visualization

from modeling.backbones.resnet import add_op


class CJY_DUAL_BACKPROPAGATION():
    """
    说明：dual 包括 gradient 和 bias

    contribution = gradient * activation + bias

    比dual backpropagation写的更好
    """
    def __init__(self, model, num_classes, target_layer):
        self.model = model
        self.num_classes = num_classes

        self.target_layer = target_layer  # 最好按forward顺序写
        self.num_target_layer = 0
        self.inter_output = []
        self.inter_gradient = []
        self.targetHookIndex = 0

        self.useGuidedReLU = True   #True  #False  # GuideBackPropagation的变体
        self.guidedReLUstate = 0    # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上
        self.num_relu_layers = 0
        self.relu_output = []
        self.relu_current_index = 0  #后续设定为len(relu_input)
        self.stem_relu_index_list = []

        self.useGuidedMAXPOOL = False  # True  #False  # GuideBackPropagation的变体
        self.guidedMAXPOOLstate = 0  # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上
        self.num_maxpool_layers = 0
        self.maxpool_output = []
        self.maxpool_current_index = 0  # 后续设定为len(relu_input)
        self.stem_maxpool_index_list = []

        self.useGuidedAVGPOOL = True  # True  #False  # GuideBackPropagation的变体
        self.guidedAVGPOOLstate = 0  # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上
        self.num_avgpool_layers = 0
        self.avgpool_output = []
        self.avgpool_current_index = 0  # 后续设定为len(relu_input)
        self.stem_avgpool_index_list = []

        self.useGuidedAdaptiveAVGPOOL = True  # True  #False  # GuideBackPropagation的变体
        self.guidedAdaptiveAVGPOOLstate = 0  # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上
        self.num_adaptive_avgpool_layers = 0
        self.adaptive_avgpool_input = []
        self.adaptive_avgpool_current_index = 0  # 后续设定为len(relu_input)
        self.stem_adaptive_avgpool_index_list = []

        self.useGuidedLINEAR = True#False  # True  # True#False  # GuideBackPropagation的变体  #只适用于前置为relu的linear，保证linear的输入为非负
        self.guidedLINEARstate = 0
        self.num_linear_layers = 0
        self.linear_input = []
        self.linear_current_index = 0

        self.useGuidedCONV = True  # True  # True#False  # GuideBackPropagation的变体
        self.guidedCONVstate = 0
        self.num_conv_layers = 0
        self.conv_input = []
        self.conv_current_index = 0

        self.useGuidedBN = True    #True  # True#False  # GuideBackPropagation的变体
        self.guidedBNstate = 0
        self.num_bn_layers = 0
        self.bn_input = []
        self.bn_current_index = 0

        self.useGuidedADD = True    #True  # True#False  # GuideBackPropagation的变体
        self.guidedADDstate = 0
        self.num_add_layers = 0
        self.add_output = []
        self.add_current_index = 0

        self.firstCAM = 1

        self.reservePos = False  #True

        self.normFlag = True

        self.multiply_input = 2

        self.bias_back_type = 2  #1,2,3

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
                    #print(module_name)
                    if tl == "": # 对输入图像进行hook
                        module.register_forward_hook(self.reserve_input_for_firstLayer_hook_fn)
                        self.num_target_layer = self.num_target_layer + 1
                        break
                    if module_name == tl:
                        print("Visualization Hook on ", module_name)
                        module.register_forward_hook(self.reserve_output_for_targetLayer_hook_fn)
                        #module.register_backward_hook(self.backward_hook_fn_for_targetLayer)    #不以backward求取gcam了，why，因为这种回传会在模型中保存梯度，然后再清零会出问题
                        self.num_target_layer = self.num_target_layer + 1
                        break
            if self.num_target_layer != len(self.target_layer):
                raise Exception("Target Hook Num can not match Target Layer Num")
        else:
            raise Exception("Without target layer can not generate Visualization")

        # 2.Set Guided-Backpropagation Hook
        if self.useGuidedReLU == True:
            print("Set GuidedBP Hook on ReLU")
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.ReLU) == True and "segmenter" not in module_name:
                    if "densenet" in self.model.base_name and "denseblock" not in module_name:
                        self.stem_relu_index_list.append(self.num_relu_layers)
                        #print("Stem ReLU:{}".format(module_name))
                    elif "resnet" in self.model.base_name and "relu1" not in module_name and "relu2" not in module_name:
                        self.stem_relu_index_list.append(self.num_relu_layers)
                        #print("Stem ReLU:{}".format(module_name))
                    elif "vgg" in self.model.base_name:
                        self.stem_relu_index_list.append(self.num_relu_layers)
                        #print("Stem ReLU:{}".format(module_name))
                    self.num_relu_layers = self.num_relu_layers + 1
                    module.register_forward_hook(self.relu_forward_hook_fn)
                    module.register_backward_hook(self.relu_backward_hook_fn)

        if self.useGuidedMAXPOOL == True:
            print("Set GuidedBP Hook on MAXPOOL")  #MaxPool也算非线性吧
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.MaxPool2d) == True and "segmenter" not in module_name:
                    self.stem_maxpool_index_list.append(self.num_maxpool_layers)
                    print("Stem MAXPOOL:{}".format(module_name))
                    self.num_maxpool_layers = self.num_maxpool_layers + 1
                    module.register_forward_hook(self.maxpool_forward_hook_fn)
                    module.register_backward_hook(self.maxpool_backward_hook_fn)

        if self.useGuidedAVGPOOL == True:
            print("Set GuidedBP Hook on AVGPOOL")  #MaxPool也算非线性吧
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.AvgPool2d) == True and "segmenter" not in module_name:
                    self.stem_avgpool_index_list.append(self.num_avgpool_layers)
                    print("Stem AVGPOOL:{}".format(module_name))
                    self.num_avgpool_layers = self.num_avgpool_layers + 1
                    module.register_forward_hook(self.avgpool_forward_hook_fn)
                    module.register_backward_hook(self.avgpool_backward_hook_fn)


        if self.useGuidedAdaptiveAVGPOOL == True:
            print("Set GuidedBP Hook on AdaptiveAVGPOOL")  #MaxPool也算非线性吧
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.AdaptiveAvgPool2d) == True and "segmenter" not in module_name:
                    self.stem_adaptive_avgpool_index_list.append(self.num_adaptive_avgpool_layers)
                    print("Stem AdaptiveAVGPOOL:{}".format(module_name))
                    self.num_adaptive_avgpool_layers = self.num_adaptive_avgpool_layers + 1
                    module.register_forward_hook(self.adaptive_avgpool_forward_hook_fn)
                    module.register_backward_hook(self.adaptive_avgpool_backward_hook_fn)


        if self.useGuidedCONV == True:
            print("Set GuidedBP Hook on CONV")
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d) == True and "segmenter" not in module_name:
                    module.register_forward_hook(self.conv_forward_hook_fn)
                    module.register_backward_hook(self.conv_backward_hook_fn)
                    self.num_conv_layers = self.num_conv_layers + 1

        if self.useGuidedLINEAR == True:
            print("Set GuidedBP Hook on LINEAR")
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) == True and "segmenter" not in module_name:
                    module.register_forward_hook(self.linear_forward_hook_fn)
                    module.register_backward_hook(self.linear_backward_hook_fn)
                    self.num_linear_layers = self.num_linear_layers + 1


        if self.useGuidedBN == True:
            print("Set GuidedBP Hook on BN")
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) == True and "segmenter" not in module_name:
                    module.register_backward_hook(self.bn_backward_hook_fn)
                    module.register_forward_hook(self.bn_forward_hook_fn)
                    self.num_bn_layers = self.num_bn_layers + 1

        if self.useGuidedADD == True:
            print("Set GuidedBP Hook on ADD")
            for module_name, module in model.named_modules():
                if isinstance(module, add_op) == True and "segmenter" not in module_name:
                    module.register_backward_hook(self.add_backward_hook_fn)
                    module.register_forward_hook(self.add_forward_hook_fn)
                    self.num_add_layers = self.num_add_layers + 1


    # Hook Function
    def reserve_input_for_firstLayer_hook_fn(self, module, input, output):
        # 为了避免多次forward，保存多个特征，所以通过计数完成置零操作
        if self.targetHookIndex % self.num_target_layer == 0:
            self.targetHookIndex = 0
            self.inter_output.clear()
            self.inter_gradient.clear()
        #input[0].requires_grad_(True)   # 在这里改input的grad好像没用；只能在forward之前更改
        self.inter_output.append(input[0])
        self.targetHookIndex = self.targetHookIndex + 1

    def reserve_output_for_targetLayer_hook_fn(self, module, input, output):
        # 为了避免多次forward，保存多个特征，所以通过计数完成置零操作
        if self.targetHookIndex % self.num_target_layer == 0:
            self.targetHookIndex = 0
            self.inter_output.clear()
            self.inter_gradient.clear()
        self.inter_output.append(output)
        self.targetHookIndex = self.targetHookIndex + 1

    def linear_forward_hook_fn(self, module, input, output):
        if self.linear_current_index == 0:
            self.linear_input.clear()
        self.linear_input.append(input[0])
        self.linear_current_index = self.linear_current_index + 1
        if self.linear_current_index % self.num_linear_layers == 0:
            self.linear_current_index = 0

    def linear_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedLINEARstate == True:
            self.linear_input_obtain_index = self.linear_input_obtain_index - 1
            linear_input = self.linear_input[self.linear_input_obtain_index]

            num_sub_batch = linear_input.shape[0] // self.multiply_input
            linear_in_sub = [linear_input[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_in_sub = [grad_in[1][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            grad_input = grad_in_sub[0]
            grad_output = grad_out_sub[0]
            bias_output = grad_out_sub[1]
            bias_current = module.bias.unsqueeze(0).expand_as(grad_output) if module.bias is not None else 0
            bias_overall = bias_output + bias_current * grad_output

            # new_bias_input计算
            if self.bias_back_type == 1:
                # 1
                new_weight = torch.ones_like(module.weight)  # module.weight
                new_weight = new_weight / (torch.sum(new_weight, dim=1, keepdim=True))
                bias_input = torch.nn.functional.linear(bias_overall, new_weight.permute(1, 0))
            elif self.bias_back_type == 2:
                # 2
                new_weight = torch.ones_like(module.weight)
                # x为0的点为死点，不将bias分给这种点
                activation_map = linear_in_sub[0].ne(0).float()
                activation_num_map = torch.nn.functional.linear(activation_map, new_weight)  # 计算非死点个数之和
                x_nonzero = (activation_num_map).ne(0).float()
                y = bias_overall / (activation_num_map + (1 - x_nonzero)) * x_nonzero
                z = torch.nn.functional.linear(y, new_weight.permute(1, 0))
                bias_input = z * activation_map
            elif self.bias_back_type == 3:
                # 3
                new_weight = module.weight.relu()
                x = torch.nn.functional.linear(linear_in_sub[0], new_weight)
                x_nonzero = x.ne(0).float()
                y = bias_overall / (x + (1 - x_nonzero)) * x_nonzero
                z = torch.nn.functional.linear(y, new_weight.permute(1, 0))
                bias_input = linear_in_sub[0] * z

            new_grad_in_sub = [grad_input, bias_input]
            new_grad_in = torch.cat(new_grad_in_sub, dim=0)

            return (grad_in[0], new_grad_in, grad_in[2])  # bias input weight


    def conv_forward_hook_fn(self, module, input, output):
        if self.conv_current_index == 0:
            self.conv_input.clear()
        self.conv_input.append(input[0])
        self.conv_current_index = self.conv_current_index + 1
        if self.conv_current_index % self.num_conv_layers == 0:
            self.conv_current_index = 0

    def conv_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedCONVstate == True:
            self.conv_input_obtain_index = self.conv_input_obtain_index - 1
            conv_input = self.conv_input[self.conv_input_obtain_index]

            num_sub_batch = conv_input.shape[0] // self.multiply_input
            conv_in_sub = [conv_input[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            grad_input = grad_in_sub[0]
            grad_output = grad_out_sub[0]
            bias_output = grad_out_sub[1]
            bias_current = module.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(grad_output) if module.bias is not None else 0
            bias_overall = bias_output + bias_current * grad_output

            # 0. preparation
            new_padding = (module.kernel_size[0] - module.padding[0] - 1, module.kernel_size[1] - module.padding[1] - 1)
            output_size = (grad_out[0].shape[3] - 1) * module.stride[0] - 2 * new_padding[0] + module.dilation[0] * (module.kernel_size[0] - 1) + 1
            output_padding = grad_in[0].shape[3] - output_size

            # new_bias_input计算
            if self.bias_back_type == 1:
                # 1
                new_weight = torch.ones_like(module.weight)  # module.weight
                new_weight = new_weight / (
                    new_weight.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True))
                bias_input = torch.nn.functional.conv_transpose2d(bias_overall, new_weight, stride=module.stride,
                                                                  padding=new_padding, output_padding=output_padding)
            elif self.bias_back_type == 2:
                # 2
                new_weight = torch.ones_like(module.weight)
                # x为0的点为死点，不将bias分给这种点
                activation_map = conv_in_sub[0].ne(0).float()
                activation_num_map = torch.nn.functional.conv2d(activation_map, new_weight, stride=module.stride,
                                                                padding=module.padding)  # 计算非死点个数之和
                x_nonzero = (activation_num_map).ne(0).float()
                y = bias_overall / (activation_num_map + (1 - x_nonzero)) * x_nonzero
                z = torch.nn.functional.conv_transpose2d(y, new_weight, stride=module.stride, padding=new_padding,
                                                         output_padding=output_padding)
                bias_input = z * activation_map
            elif self.bias_back_type == 3:
                # 3
                new_weight = module.weight.relu()
                x = torch.nn.functional.conv2d(conv_in_sub[0], new_weight, stride=module.stride, padding=module.padding)
                x_nonzero = x.ne(0).float()
                y = bias_overall / (x + (1 - x_nonzero)) * x_nonzero  # 文章中并没有说应该怎么处理分母为0的情况
                z = torch.nn.functional.conv_transpose2d(y, new_weight, stride=module.stride, padding=new_padding,
                                                         output_padding=output_padding)
                bias_input = conv_in_sub[0] * z

            self.rest = self.rest + bias_overall.sum() - bias_input.sum()

            new_grad_in_sub = [grad_input, bias_input]
            new_grad_in = torch.cat(new_grad_in_sub, dim=0)

            return (new_grad_in, grad_in[1], grad_in[2])


    def relu_forward_hook_fn(self, module, input, output):
        if self.relu_current_index == 0:
            self.relu_output.clear()
        self.relu_output.append(output)
        self.relu_current_index = self.relu_current_index + 1
        if self.relu_current_index % self.num_relu_layers == 0:
            self.relu_current_index = 0

    def relu_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedReLUstate == True:
            self.relu_output_obtain_index = self.relu_output_obtain_index - 1
            relu_output = self.relu_output[self.relu_output_obtain_index]

            num_sub_batch = relu_output.shape[0]//self.multiply_input
            relu_out_sub = [relu_output[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            if self.bias_back_type == 1:
                new_grad_in_sub = [grad_in_sub[0], grad_out_sub[1]]
                new_grad_in = torch.cat(new_grad_in_sub, dim=0)
            elif self.bias_back_type == 2:
                new_grad_in = grad_in[0]

            """
            if grad_out[0].ndimension() == 4:
                cam = self.GenerateCAM(relu_output, grad_out[0]).gt(0).float()
                new_grad_in = new_grad_in * cam
            #"""

            return (new_grad_in,)


    def maxpool_forward_hook_fn(self, module, input, output):
        if self.maxpool_current_index == 0:
            self.maxpool_output.clear()
        self.maxpool_output.append(output)
        self.maxpool_current_index = self.maxpool_current_index + 1
        if self.maxpool_current_index % self.num_maxpool_layers == 0:
            self.maxpool_current_index = 0

    def maxpool_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedMAXPOOLstate == True:
            self.maxpool_output_obtain_index = self.maxpool_output_obtain_index - 1
            maxpool_output = self.maxpool_output[self.maxpool_output_obtain_index]

            new_grad_in = grad_in[0]
            return (new_grad_in,)

    def avgpool_forward_hook_fn(self, module, input, output):
        if self.avgpool_current_index == 0:
            self.avgpool_output.clear()
        self.avgpool_output.append(input[0])
        self.avgpool_current_index = self.avgpool_current_index + 1
        if self.avgpool_current_index % self.num_avgpool_layers == 0:
            self.avgpool_current_index = 0

    def avgpool_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedAVGPOOLstate == True:
            self.avgpool_output_obtain_index = self.avgpool_output_obtain_index - 1
            avgpool_output = self.avgpool_output[self.avgpool_output_obtain_index]

            num_sub_batch = avgpool_output.shape[0] // self.multiply_input
            avgpool_out_sub = [avgpool_output[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            bias_input = grad_in_sub[1]

            if self.bias_back_type == 1:
                new_grad_in = grad_in[0]
            elif self.bias_back_type == 2:
                num_solo = module.num_solo_activation_neuron
                num_pool = module.num_pool_activation_neuron
                ratio = num_solo/ num_pool

                ratio_sub = [ratio[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

                new_bias = bias_input * ratio_sub[0]

                new_grad_in_sub = [grad_in_sub[0], new_bias]
                new_grad_in = torch.cat(new_grad_in_sub, dim=0)

            return (new_grad_in,)

    def adaptive_avgpool_forward_hook_fn(self, module, input, output):
        if self.adaptive_avgpool_current_index == 0:
            self.adaptive_avgpool_input.clear()
        self.adaptive_avgpool_input.append(input[0])
        self.adaptive_avgpool_current_index = self.adaptive_avgpool_current_index + 1
        if self.adaptive_avgpool_current_index % self.num_adaptive_avgpool_layers == 0:
            self.adaptive_avgpool_current_index = 0

    def adaptive_avgpool_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedAVGPOOLstate == True:
            self.adaptive_avgpool_input_obtain_index = self.adaptive_avgpool_input_obtain_index - 1
            adaptive_avgpool_input = self.adaptive_avgpool_input[self.adaptive_avgpool_input_obtain_index]

            if self.bias_back_type == 1:
                if grad_in[0].ndimension() != 4:
                    return grad_in

                num_sub_batch = adaptive_avgpool_input.shape[0] // self.multiply_input
                adaptive_avgpool_in_sub = [adaptive_avgpool_input[i * num_sub_batch: (i + 1) * num_sub_batch] for i in
                                            range(self.multiply_input)]
                grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in
                                range(self.multiply_input)]
                grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in
                               range(self.multiply_input)]

                bias_overall = grad_out_sub[1]

                bias_overall_sum = bias_overall.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

                new_bias = torch.ones_like(grad_in_sub[1]) * bias_overall_sum / (
                            grad_in_sub[1].shape[2] * grad_in_sub[1].shape[3])

                new_grad_in_sub = [grad_in_sub[0], new_bias]
                new_grad_in = torch.cat(new_grad_in_sub, dim=0)

                self.rest = self.rest + bias_overall.sum() - new_bias.sum()

                return (new_grad_in,)

            elif self.bias_back_type == 2:
                num_sub_batch = adaptive_avgpool_input.shape[0] // self.multiply_input
                adaptive_avgpool_in_sub = [adaptive_avgpool_input[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
                grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

                # 此处将所有bias求和再重新分配
                bias_overall = grad_out_sub[1]
                bias_overall_sum = bias_overall.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

                activation_map = adaptive_avgpool_in_sub[0].gt(0).float()
                activation_map_sum = activation_map.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

                # activation map 全体与非0个数的比例
                ratio = activation_map.shape[1] * activation_map.shape[2] * activation_map.shape[3] / activation_map_sum

                if grad_in[0].ndimension() == 1:
                    channels = grad_out[0].shape[1]
                    grad_in_sub = [grad_in[0][i * channels: (i + 1) * channels] for i in range(self.multiply_input)]

                    grad_in_sub1 = grad_in_sub[1].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).view(-1, channels, 1, 1)
                    new_bias = torch.ones_like(grad_in_sub1) * bias_overall_sum * ratio / grad_in_sub[1].shape[0]

                    new_bias = new_bias.view_as(grad_in_sub[1])

                    new_grad_in_sub = [grad_in_sub[0], new_bias]
                    new_grad_in = torch.cat(new_grad_in_sub, dim=0)
                    # 注：此时的new_bias必须在后面avg-pool backward后与activation_map相乘才好

                else:
                    grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

                    new_bias = torch.ones_like(grad_in_sub[1]) * bias_overall_sum * ratio/ (
                            grad_in_sub[1].shape[1] * grad_in_sub[1].shape[2] * grad_in_sub[1].shape[3])

                    new_bias = new_bias * activation_map

                    new_grad_in_sub = [grad_in_sub[0], new_bias]
                    new_grad_in = torch.cat(new_grad_in_sub, dim=0)

                #self.rest = self.rest + bias_overall.sum() - new_bias.sum()

                return (new_grad_in,)


    def bn_forward_hook_fn(self, module, input, output):
        """
        eps = module.eps
        mean = module.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        var = module.running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        weight = module.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = module.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        output = (input[0] - mean) / (var+eps).sqrt() * weight + bias
        """
        if self.bn_current_index == 0:
            self.bn_input.clear()
        self.bn_input.append(input[0])
        self.bn_current_index = self.bn_current_index + 1
        if self.bn_current_index % self.num_bn_layers == 0:
            self.bn_current_index = 0

    def bn_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedBNstate == True:
            self.bn_input_obtain_index = self.bn_input_obtain_index - 1
            bn_input = self.bn_input[self.bn_input_obtain_index]

            eps = module.eps
            mean = module.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            var = module.running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            weight = module.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            bias = module.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            num_sub_batch = bn_input.shape[0] // self.multiply_input
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            grad_input = grad_in_sub[0]
            grad_output = grad_out_sub[0]
            bias_output = grad_out_sub[1]
            bias_current = - (mean * weight) / (var + eps).sqrt() + bias
            bias_overall = bias_output + bias_current * grad_output

            # new_bias_input计算
            bias_input = bias_overall

            new_grad_in_sub = [grad_input, bias_input]
            new_grad_in = torch.cat(new_grad_in_sub, dim=0)

            return (new_grad_in, grad_in[1], grad_in[2])


    def add_forward_hook_fn(self, module, input, output):
        if self.add_current_index == 0:
            self.add_output.clear()
        self.add_output.append(output)
        self.add_current_index = self.add_current_index + 1
        if self.add_current_index % self.num_add_layers == 0:
            self.add_current_index = 0

    def add_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedADDstate == True:
            self.add_output_obtain_index = self.add_output_obtain_index - 1
            add_output = self.add_output[self.add_output_obtain_index]

            num_sub_batch = add_output.shape[0] // self.multiply_input
            grad_out_sub = [grad_out[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            identity_grad_in_sub = [grad_in[0][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
            residual_grad_in_sub = [grad_in[1][i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

            bias_overall = grad_out_sub[1]

            if self.bias_back_type == 1:
                identity_ratio = module.num_identity_neuron / (module.num_identity_neuron + module.num_residual_neuron)
                residual_ratio = module.num_residual_neuron / (module.num_identity_neuron + module.num_residual_neuron)
            elif self.bias_back_type == 2:
                num_identity_an_sub = [module.num_identity_activation_neuron[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
                num_residual_an_sub = [module.num_residual_activation_neuron[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
                identity_ratio = num_identity_an_sub[0] / (num_identity_an_sub[0] + num_residual_an_sub[0])
                residual_ratio = num_residual_an_sub[0] / (num_identity_an_sub[0] + num_residual_an_sub[0])

            new_identity_bias = bias_overall * identity_ratio
            new_residual_bias = bias_overall * residual_ratio

            new_grad_in0 = torch.cat([identity_grad_in_sub[0], new_identity_bias], dim=0)
            new_grad_in1 = torch.cat([residual_grad_in_sub[0], new_residual_bias], dim=0)

            return (new_grad_in0, new_grad_in1)


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

        self.guidedLINEARstate = 1
        self.guidedCONVstate = 1
        self.guidedBNstate = 1
        self.guidedReLUstate = 1
        self.guidedMAXPOOLstate = 1
        self.guidedAVGPOOLstate = 1
        self.guidedAdaptiveAVGPOOLstate = 1
        self.guidedADDstate = 1

        self.firstCAM = 1
        self.linear_input_obtain_index = len(self.linear_input)
        self.conv_input_obtain_index = len(self.conv_input)
        self.bn_input_obtain_index = len(self.bn_input)
        self.relu_output_obtain_index = len(self.relu_output)
        self.maxpool_output_obtain_index = len(self.maxpool_output)
        self.avgpool_output_obtain_index = len(self.avgpool_output)
        self.adaptive_avgpool_input_obtain_index = len(self.adaptive_avgpool_input)

        self.add_output_obtain_index = len(self.add_output)

        self.rest = 0

        if self.multiply_input >= 1:
            #gcam_one_hot_labels = torch.cat([gcam_one_hot_labels] * self.multiply_input, dim=0)
            gcam_one_hot_labels = torch.cat([gcam_one_hot_labels, gcam_one_hot_labels * 0], dim=0)

        inter_gradients = torch.autograd.grad(outputs=logits, inputs=self.inter_output,
                                              grad_outputs=gcam_one_hot_labels,
                                              retain_graph=True)#, create_graph=True)   #由于显存的问题，不得已将retain_graph
        self.inter_gradient = list(inter_gradients)

        self.guidedLINEARstate = 0
        self.guidedCONVstate = 0
        self.guidedBNstate = 0
        self.guidedReLUstate = 0
        self.guidedMAXPOOLstate = 0
        self.guidedAVGPOOLstate = 0
        self.guidedAdaptiveAVGPOOLstate = 0
        self.guidedADDstate = 0


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


    # Generate Single CAM (backward)
    def GenerateCAM(self, inter_output, inter_gradient):
        # backward形式
        num_sub_batch = inter_output.shape[0] // self.multiply_input
        inter_output_sub = [inter_output[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]
        inter_gradient_sub = [inter_gradient[i * num_sub_batch: (i + 1) * num_sub_batch] for i in range(self.multiply_input)]

        inter_output = inter_output_sub[0]
        inter_gradient = inter_gradient_sub[0]
        inter_bias = inter_gradient_sub[1]

        if self.bias_back_type == 2:
            if inter_output.shape[1] != 3:
                inter_bias = inter_output.gt(0).float() * inter_bias

        gcam = torch.sum(inter_gradient * inter_output + inter_bias, dim=1, keepdim=True)

        gcam_l = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
        gcam_b = torch.sum(inter_bias, dim=1, keepdim=True)
        print("linear:{}, bias:{}, sum:{}".format(gcam_l.sum(), gcam_b.sum(), gcam.sum()))
        print("linear_max:{}, bias_max:{}, sum_max:{}".format(gcam_l.max(), gcam_b.max(), gcam.max()))

        if self.reservePos == True:
            gcam = torch.relu(gcam)

        return gcam

    # Generate Overall CAM
    def GenerateOverallCAM(self, gcam_list, input_size):
        # 多尺度下的gcam进行融合
        """
        #1.concat -  max or mean
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
        """
        #2.norm multiply
        overall_gcam = 0
        for index, gcam in enumerate(reversed(gcam_list)):
            if self.target_layer[self.num_target_layer - index - 1] == "":
                continue

            if overall_gcam is 0:
                if self.reservePos == True:
                    overall_gcam = gcam
                else:
                    overall_gcam = (gcam - 0.5).relu() * 2
            else:
                overall_gcam = torch.nn.functional.interpolate(overall_gcam, (gcam.shape[2], gcam.shape[3]), mode='bilinear')
                if self.reservePos == True:
                    overall_gcam = overall_gcam * gcam
                else:
                    overall_gcam = overall_gcam * (gcam - 0.5).relu() * 2
        overall_gcam, _ = self.gcamNormalization(overall_gcam)        
        #"""
        #"""
        #3. weighted add
        overall_gcam = 0
        for index, gcam in enumerate(reversed(gcam_list)):
            if self.target_layer[self.num_target_layer - index - 1] == "":
                continue
            if overall_gcam is 0:
                if self.reservePos == True:
                    overall_gcam = gcam
                else:
                    overall_gcam = gcam - 0.5
            else:
                overall_gcam = torch.nn.functional.interpolate(overall_gcam, (gcam.shape[2], gcam.shape[3]), mode='bilinear')
                if self.reservePos == True:
                    overall_gcam = (overall_gcam + gcam)#/2
                else:
                    overall_gcam = (overall_gcam + (gcam-0.5))#/2
        #"""

        overall_gcam = torch.nn.functional.interpolate(overall_gcam, input_size, mode='bilinear') if overall_gcam is not 0 else None


        return overall_gcam


    # Generate Visualiztions Function   # 统一用GenerateVisualiztion这个名字吧
    def GenerateVisualiztions(self, logits, labels, input_size, visual_num):
        target_layer_num = len(self.target_layer)
        self.gcam_list = []
        self.gcam_max_list = [] # 记录每个Grad-CAM的归一化最大值

        if self.multiply_input >= 1:
            visual_num = visual_num * self.multiply_input

        # obtain gradients
        self.ObtainGradient(logits, labels)

        for i in range(target_layer_num):
            # 1.获取倒数visual_num个样本的activation以及gradient
            batch_num = logits.shape[0]
            visual_num = visual_num
            inter_output = self.inter_output[i][batch_num - visual_num:batch_num]  # 此处分离节点，别人皆不分离  .detach()
            inter_gradient = self.inter_gradient[i][batch_num - visual_num:batch_num]

            # 2.生成CAM
            gcam = self.GenerateCAM(inter_output, inter_gradient)
            #print("{}: {}".format(self.target_layer[i], gcam.sum()))

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
        #print("logits:{} label:{}".format(logits[0][labels].item(), labels.item()))
        print("logits:{} rest:{} diff:{} label:{}".format(logits[0][labels].item(), self.rest.item(),
                                                          logits[0][labels].item() - self.rest.item(), labels.item()))

        # Clear Reservation
        #self.inter_output.clear()
        self.inter_gradient.clear()

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

            # 绘制一下overall_gcam
            if 0:#self.overall_gcam is not None:
                layer_name = "overall"
                visual_prefix = layer_name.replace(".", "-") + "_S{}".format(self.observation_class[j])
                if gtmasks is not None:
                    draw_visualization(imgs[j], self.overall_gcam[j], gtmasks[j], threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)
                else:
                    draw_visualization(imgs[j], self.overall_gcam[j], None, threshold, savePath, imgsName[j], label_prefix, visual_prefix, draw_flag_dict)

        return 0






