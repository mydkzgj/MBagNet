"""
Created on 2020.7.4

@author: Jiayang Chen - github.com/mydkzgj
"""

import torch
from .draw_tool import draw_visualization


class CJY_CONTRAST_GUIDED_BACKPROPAGATION():
    """
    说明：将每个特征activation的gradient分为正负两部分，可以分别观测这两部分的作用
    gcam = pos_gcam + neg_gcam 即为 pgrad-cam
    单独观察pos-gcam和neg-gcam并没有明显的类别差别
    而由于pos和neg的值越来越大，导致误差越来越大，所以前几层的gcam不再保持pgcam
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

        self.useGuidedPOOL = False  # True  #False  # GuideBackPropagation的变体
        self.guidedPOOLstate = 0  # 用于区分是进行导向反向传播还是经典反向传播，guidedBP只是用于设置hook。需要进行导向反向传播的要将self.guidedBPstate设置为1，结束后关上
        self.num_pool_layers = 0
        self.pool_output = []
        self.pool_current_index = 0  # 后续设定为len(relu_input)
        self.stem_pool_index_list = []

        self.useGuidedLINEAR = False  # True  # True#False  # GuideBackPropagation的变体  #只适用于前置为relu的linear，保证linear的输入为非负
        self.guidedLINEARstate = 0
        self.num_linear_layers = 0
        self.linear_input = []
        self.linear_current_index = 0

        self.useGuidedCONV = False  # True  # True#False  # GuideBackPropagation的变体
        self.guidedCONVstate = 0
        self.num_conv_layers = 0
        self.conv_input = []
        self.conv_current_index = 0

        self.useGuidedBN = False    #True  # True#False  # GuideBackPropagation的变体
        self.guidedBNstate = 0
        self.num_bn_layers = 0

        self.firstCAM = 1

        self.reservePos = False  #True

        self.normFlag = True

        self.double_input = True

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

        if self.useGuidedPOOL == True:
            print("Set GuidedBP Hook on POOL")  #MaxPool也算非线性吧
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.MaxPool2d) == True and "segmenter" not in module_name:
                    if "densenet" in self.model.base_name and "denseblock" not in module_name:
                        self.stem_pool_index_list.append(self.num_pool_layers)
                        print("Stem POOL:{}".format(module_name))
                    elif "resnet" in self.model.base_name and "relu1" not in module_name and "relu2" not in module_name:
                        self.stem_pool_index_list.append(self.num_pool_layers)
                        print("Stem POOL:{}".format(module_name))
                    elif "vgg" in self.model.base_name:
                        self.stem_pool_index_list.append(self.num_pool_layers)
                        print("Stem POOL:{}".format(module_name))
                    self.num_pool_layers = self.num_pool_layers + 1
                    module.register_forward_hook(self.pool_forward_hook_fn)
                    module.register_backward_hook(self.pool_backward_hook_fn)

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
                    if "resnet" in self.model.base_name and "bn3" in module_name:
                        module.register_backward_hook(self.bn_backward_hook_fn_with_zero_input)
                    else:
                        module.register_backward_hook(self.bn_backward_hook_fn)
                    #module.register_forward_hook(self.bn_forward_hook_fn)
                    self.num_bn_layers = self.num_bn_layers + 1


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

            if self.firstCAM == True:
                num_batch = linear_input.shape[0]
                grad_input = grad_in[0][0:num_batch // 2]

                new_grad_in1 = grad_input * grad_input.gt(0).float()
                new_grad_in2 = (-grad_input) * grad_input.lt(0).float()

                new_grad_in = torch.cat([new_grad_in1, new_grad_in2], dim=0)

                self.firstCAM = 0

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

            num_batch = grad_out[0].shape[0]
            pos_gard_out = grad_out[0][0:num_batch//2]
            neg_gard_out = grad_out[0][num_batch // 2:num_batch]

            # prepare for transposed conv
            new_padding = (module.kernel_size[0] - module.padding[0] - 1, module.kernel_size[1] - module.padding[1] - 1)
            output_size = (grad_out[0].shape[3] - 1) * module.stride[0] - 2 * new_padding[0] + module.dilation[0] * (
                    module.kernel_size[0] - 1) + 1
            output_padding = grad_in[0].shape[3] - output_size


            pos_weight = module.weight.relu()
            neg_weight = -(-module.weight).relu()

            pp_new_grad_in = torch.nn.functional.conv_transpose2d(pos_gard_out, pos_weight, stride=module.stride,
                                                               padding=new_padding, output_padding=output_padding)

            pn_new_grad_in = torch.nn.functional.conv_transpose2d(pos_gard_out, neg_weight, stride=module.stride,
                                                               padding=new_padding, output_padding=output_padding)

            np_new_grad_in = torch.nn.functional.conv_transpose2d(neg_gard_out, pos_weight, stride=module.stride,
                                                               padding=new_padding, output_padding=output_padding)

            nn_new_grad_in = torch.nn.functional.conv_transpose2d(neg_gard_out, neg_weight, stride=module.stride,
                                                               padding=new_padding, output_padding=output_padding)

            pos_new_grad_in = pp_new_grad_in + nn_new_grad_in
            neg_new_grad_in = pn_new_grad_in + np_new_grad_in

            new_grad_in = torch.cat([pos_new_grad_in, neg_new_grad_in], dim=0)

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

            pos_grad_out = grad_out[0].gt(0).float()
            new_grad_in = pos_grad_out * grad_in[0]

            """
            if self.firstCAM == True:
                num_batch = relu_output.shape[0]
                grad_input = grad_in[0][0:num_batch // 2]

                new_grad_in1 = grad_input * grad_input.gt(0).float()
                new_grad_in2 = (-grad_input) * grad_input.lt(0).float()

                new_grad_in = torch.cat([new_grad_in1, new_grad_in2], dim=0)

                self.firstCAM = 0
            """

            return (new_grad_in,)


    def pool_forward_hook_fn(self, module, input, output):
        if self.pool_current_index == 0:
            self.pool_output.clear()
        self.pool_output.append(output)
        self.pool_current_index = self.pool_current_index + 1
        if self.pool_current_index % self.num_pool_layers == 0:
            self.pool_current_index = 0

    def pool_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedPOOLstate == True:
            self.pool_output_obtain_index = self.pool_output_obtain_index - 1
            pool_output = self.pool_output[self.pool_output_obtain_index]

            new_grad_in = grad_in[0]
            return (new_grad_in,)


    def bn_forward_hook_fn(self, module, input, output):
        eps = module.eps
        mean = module.running_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        var = module.running_var.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        weight = module.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        bias = module.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        output = (input[0] - mean) / (var+eps).sqrt() * weight + bias

    def bn_backward_hook_fn(self, module, grad_in, grad_out):
        if self.guidedBNstate == True:
            new_grad_in = grad_in[0]
            return (new_grad_in, grad_in[1], grad_in[2])

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

        #self.bn_weight_reserve_state = 1
        self.guidedReLUstate = 1  # 是否开启guidedBP
        self.guidedPOOLstate = 1
        self.guidedCONVstate = 1
        self.guidedLINEARstate = 1
        self.guidedBNstate = 1

        self.firstCAM = 1
        self.relu_output_obtain_index = len(self.relu_output)
        self.pool_output_obtain_index = len(self.pool_output)
        self.conv_input_obtain_index = len(self.conv_input)
        self.linear_input_obtain_index = len(self.linear_input)

        if self.double_input == True:
            gcam_one_hot_labels = torch.cat([gcam_one_hot_labels, gcam_one_hot_labels*0], dim=0)

        inter_gradients = torch.autograd.grad(outputs=logits, inputs=self.inter_output,
                                              grad_outputs=gcam_one_hot_labels,
                                              retain_graph=True)#, create_graph=True)   #由于显存的问题，不得已将retain_graph
        self.inter_gradient = list(inter_gradients)

        self.guidedBNstate = 0
        self.guidedLINEARstate = 0
        self.guidedCONVstate = 0
        self.guidedPOOLstate = 0
        self.guidedReLUstate = 0


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
        """
        avg_gradient = torch.nn.functional.adaptive_max_pool2d(inter_gradient, 1).squeeze(-1).squeeze(-1).squeeze(0)
        m = torch.sort(avg_gradient)
        i = m[0][-1]
        v = m[1][-1]
        gcam = inter_output[:, v:v+1]#torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
        """

        #gcam = torch.sum(inter_output * torch.nn.functional.adaptive_avg_pool2d(inter_gradient, 1), dim=1, keepdim=True)

        #gcam = torch.sum(inter_output, dim=1, keepdim=True)
        #gcam = torch.sum(torch.nn.functional.adaptive_avg_pool2d(inter_gradient, 1) * inter_output, dim=1, keepdim=True)
        #gcam = inter_output[:, 435:436,]
        #gcam = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
        #gcam = gcam * (gcam.shape[-1] * gcam.shape[-2])  # 如此，形式上与最后一层计算的gcam量级就相同了  （由于最后loss使用mean，所以此处就不mean了）
        gcam_all = torch.sum(inter_gradient * inter_output, dim=1, keepdim=True)
        num_batch = gcam_all.shape[0]
        pos_gcam = gcam_all[0: num_batch//2]
        neg_gcam = gcam_all[num_batch//2: num_batch]
        gcam = pos_gcam - neg_gcam
        return gcam

    # Generate Overall CAM
    def GenerateOverallCAM(self, gcam_list, input_size):
        """
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
        """
        overall_gcam = 0
        for index, gcam in enumerate(reversed(gcam_list)):
            if self.target_layer[self.num_target_layer - index - 1]=="":
                continue

            if overall_gcam is 0:
                if self.reservePos == True:
                    overall_gcam = gcam
                else:
                    overall_gcam = gcam - 0.5
            else:
                overall_gcam = torch.nn.functional.interpolate(overall_gcam, (gcam.shape[2], gcam.shape[3]), mode='bilinear')
                if self.reservePos == True:
                    overall_gcam = overall_gcam * gcam
                else:
                    overall_gcam = overall_gcam * (gcam-0.5)
        overall_gcam = torch.nn.functional.interpolate(overall_gcam, input_size, mode='bilinear')
        #"""

        overall_gcam = 0
        for index, gcam in enumerate(reversed(gcam_list)):
            if self.target_layer[self.num_target_layer - index - 1]=="":
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
        if overall_gcam is not 0:
            overall_gcam = torch.nn.functional.interpolate(overall_gcam, input_size, mode='bilinear')
        else:
            overall_gcam = None

        return overall_gcam


    # Generate Visualiztions Function   # 统一用GenerateVisualiztion这个名字吧
    def GenerateVisualiztions(self, logits, labels, input_size, visual_num):
        target_layer_num = len(self.target_layer)
        self.gcam_list = []
        self.gcam_max_list = [] # 记录每个Grad-CAM的归一化最大值

        if self.double_input == True:
            visual_ratio = 2
        else:
            visual_ratio = 1

        # obtain gradients
        self.ObtainGradient(logits, labels)

        for i in range(target_layer_num):
            # 1.获取倒数visual_num个样本的activation以及gradient
            batch_num = logits.shape[0]
            visual_num = visual_num * visual_ratio
            inter_output = self.inter_output[i][batch_num - visual_num:batch_num]  # 此处分离节点，别人皆不分离  .detach()
            inter_gradient = self.inter_gradient[i][batch_num - visual_num:batch_num]

            # 2.生成CAM
            gcam = self.GenerateCAM(inter_output, inter_gradient)
            print("{}: {}".format(self.target_layer[i], gcam.sum()))

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
        print("logits:{} label:{}".format(logits[0][labels].item(), labels.item()))

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






