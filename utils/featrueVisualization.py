
#做一个可视化
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

save_img_index = 0
savePath = "D:/MIP/Experiment/MBagNet/work_space/heatmap/"

def map_visualization(map_tensor):
    map = map_tensor.cpu().detach().numpy()
    show = map
    a = np.max(show)
    if False:#a != 0:
        show =show/a
    #cv.imshow("1", show/a)
    #cv.waitKey(0)
    return show

def one_map_visualization(map_tensor):
    map = map_tensor.cpu().detach().numpy()
    show = map
    a = np.max(show)
    cv.imshow("1", show/a)
    cv.waitKey(0)
    return show/a

def maps_show(weight, h, w):
    # """h, w
    if 1:  # self.in_channels == 512:
        i = 0
        j = 0
        s = 4

        while i < h:
            while j < w:
                index = i * w + j
                print(i, j)
                context_mask = weight[0][index].view(h, w)
                map = map_visualization(context_mask)
                if j == 0:
                    hmap = map
                else:
                    hmap = cv.hconcat([hmap, map])
                j += s
            j = 0
            if i == 0:
                wholemap = hmap
            else:
                wholemap = cv.vconcat([wholemap, hmap])
            i += s
        # a = np.max(wholemap)
        cv.imshow("1", wholemap)
        cv.waitKey(0)
    # """

def maps_show2(weight, n):
    # """h, w
    if 1:  # self.in_channels == 512:
        batch_index = 0   #

        i = 0
        j = 0
        s = 1

        h = 8
        w = 8


        while i < h:
            while j < w:
                index = i * w + j
                if index >= n:
                    break
                print(i, j)
                context_mask = weight[index][0]
                map = map_visualization(context_mask)
                if j == 0:
                    hmap = map
                else:
                    hmap = cv.hconcat([hmap, map])
                j += s
            if index >= n:
                break
            j = 0
            if i == 0:
                wholemap = hmap
            else:
                wholemap = cv.vconcat([wholemap, hmap])
            i += s

        # a = np.max(wholemap)
        cv.imshow("batch_attention", wholemap)
        cv.waitKey(0)
    # """

def maps_show3(weight, n_c):
    # """h, w
    if 1:  # self.in_channels == 512:
        batch_index = 0   #

        i = 0
        j = 0
        s = 1

        h = 2#16
        w = n_c//h


        while i < h:
            while j < w:
                index = i * w + j
                if index >= n_c:
                    break
                #print(i, j)
                context_mask = weight[0][index]
                map = map_visualization(context_mask)
                if j == 0:
                    hmap = map
                else:
                    hmap = cv.hconcat([hmap, map])
                j += s
            if index >= n_c:
                break
            j = 0
            if i == 0:
                wholemap = hmap
            else:
                wholemap = cv.vconcat([wholemap, hmap])
            i += s

        # a = np.max(wholemap)
        cv.imshow("batch_attention", wholemap)
        cv.waitKey(0)
    # """


import torch
# CJY at 2019.12.5  用于观察densenet（改）之后的参数权值分布
"""
def showWeight(model):
    #cjy  at 2019.11.29
    params_dict = {}
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        params_dict[name] = parameters.detach().cpu()#.numpy()
    # print(len(params_dict))
    import numpy as np
    db = []
    db.append(np.zeros([7,8]))
    db.append(np.zeros([13,14]))
    db.append(np.zeros([25, 26]))
    db.append(np.zeros([17, 18]))

    denlayer_num = [6,12,24,16]

    grow_rate = 16
    for name in params_dict.keys():
        if "conv1" in name:
            sub = name.split(".")
            denseblock_index = int(sub[2].replace("denseblock",""))
            denselayer_index = int(sub[3].replace("denselayer",""))
            weight = params_dict[name]
            in_channels = weight.shape[1]
            for i in range(denselayer_index):
                if i == denselayer_index-1:
                    sub_weight = weight[:, 0:(in_channels - grow_rate * i)]
                else:
                    sub_weight = weight[:,(in_channels-grow_rate*(i+1)):(in_channels-grow_rate*i)]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum/(sub_weight.shape[1]*sub_weight.shape[0])
                print(sub_mean)
                db[denseblock_index-1][denselayer_index-i-1][denselayer_index] = sub_mean.item()
            print(sub)
        elif "transition" in name and "conv" in name:
            sub = name.split(".")
            transition_index = int(sub[2].replace("transition",""))
            weight = params_dict[name]
            in_channels = weight.shape[1]

            denselayer_index = denlayer_num[transition_index-1]+1
            for i in range(denselayer_index):
                if i == denselayer_index-1:
                    sub_weight = weight[:, 0:(in_channels - grow_rate * i)]
                else:
                    sub_weight = weight[:,(in_channels-grow_rate*(i+1)):(in_channels-grow_rate*i)]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum/(sub_weight.shape[1]*sub_weight.shape[0])
                print(sub_mean)
                db[transition_index-1][denselayer_index-i-1][denselayer_index] = sub_mean.item()
            print(sub)
        elif "classifier.weight" in name:
            sub = name.split(".")
            transition_index = 4
            weight = params_dict[name]
            in_channels = weight.shape[1]

            denselayer_index = denlayer_num[transition_index-1]+1
            for i in range(denselayer_index):
                if i == denselayer_index-1:
                    sub_weight = weight[:, 0:(in_channels - grow_rate * i)]
                else:
                    sub_weight = weight[:,(in_channels-grow_rate*(i+1)):(in_channels-grow_rate*i)]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum/(sub_weight.shape[1]*sub_weight.shape[0])
                print(sub_mean)
                db[transition_index-1][denselayer_index-i-1][denselayer_index] = sub_mean.item()
            print(sub)

    import seaborn as sns
    import matplotlib.pyplot as plt
    for matrix in db:
        sumValue = np.max(matrix,axis=0)+1E-12
        ax = sns.heatmap(matrix/sumValue)
        plt.show()
    print("finish")
"""


def showWeight(model):
    #cjy  at 2019.11.29
    params_dict = {}
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        params_dict[name] = parameters.detach().cpu()#.numpy()
    # print(len(params_dict))
    import numpy as np
    db = []
    db.append(np.zeros([7,8]))
    db.append(np.zeros([13,14]))
    db.append(np.zeros([25, 26]))
    db.append(np.zeros([17, 18]))
    db.append(np.zeros([59, 1]))

    denlayer_num = [6,12,24,16]

    grow_rate = [64,32,16,8]
    for name in params_dict.keys():
        if "conv1" in name:
            sub = name.split(".")
            denseblock_index = int(sub[2].replace("denseblock",""))
            denselayer_index = int(sub[3].replace("denselayer",""))
            weight = params_dict[name]
            in_channels = weight.shape[1]
            for i in range(denselayer_index):
                if i == denselayer_index-1:
                    sub_weight = weight[:, 0:(in_channels - grow_rate[denseblock_index-1] * i)]
                else:
                    sub_weight = weight[:,(in_channels-grow_rate[denseblock_index-1]*(i+1)):(in_channels-grow_rate[denseblock_index-1]*i)]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum/(sub_weight.shape[1]*sub_weight.shape[0])
                print(sub_mean)
                db[denseblock_index-1][denselayer_index-i-1][denselayer_index] = sub_mean.item()
            print(sub)
        elif "transition" in name and "conv" in name:
            sub = name.split(".")
            transition_index = int(sub[2].replace("transition",""))
            weight = params_dict[name]
            out_channels = weight.shape[0]

            denselayer_index = denlayer_num[transition_index-1]+1
            for i in range(denselayer_index):
                if i == denselayer_index-1:
                    sub_weight = weight[0:(out_channels - grow_rate[denseblock_index-1]//2 * i)]
                else:
                    sub_weight = weight[(out_channels-grow_rate[denseblock_index-1]//2 *(i+1)):(out_channels-grow_rate[denseblock_index-1]//2 *i)]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum/(sub_weight.shape[1]*sub_weight.shape[0])
                print(sub_mean)
                db[transition_index-1][denselayer_index-i-1][denselayer_index] = sub_mean.item()
            print(sub)
        elif "rf_classifier.weight" in name:  #group
            sub = name.split(".")
            transition_index = 4
            weight = params_dict[name]

            num_receptive_field = 59
            num_channel_per_rf = weight.shape[1]

            rest_weight = 0
            for i in range(num_receptive_field):
                sub_weight = weight[i*num_channel_per_rf:(i+1)*num_channel_per_rf]
                sub_sum = torch.sum(torch.abs(sub_weight))
                sub_mean = sub_sum / (sub_weight.shape[1] * sub_weight.shape[0])
                print(sub_mean)
                if num_receptive_field - i <= denlayer_num[transition_index-1]:
                    index = denlayer_num[transition_index-1] - num_receptive_field + i
                    db[transition_index-1][index+1][-1] = sub_mean.item()
                else:
                    rest_weight = rest_weight + sub_mean.item()
                db[4][i][0] = sub_mean.item()
            rest_weight = rest_weight/(num_receptive_field - denlayer_num[transition_index-1])
            db[transition_index-1][0][-1] = rest_weight

    import seaborn as sns
    import matplotlib.pyplot as plt
    for matrix in db:
        sumValue = np.max(matrix,axis=0)+1E-12
        ax = sns.heatmap(matrix/sumValue)
        plt.show()
    print("finish")

#CJY BagNet 可视化
def showBagNetEvidence():
    from utils import bagnet_utils as bu
    bu.show(model, imgs[0].unsqueeze(0).cpu().detach().numpy(), labels[0].item(), 9)

# CJY Grad-CAM可视化
def showGradCAM(model, imgs, labels, target_layers, mask=None):
    # CJY 注：Grad-CAM由于要求导，所以不能放在with torch.no_grad()里面
    # visualization
    from utils.visualisation.gradcam import GradCam
    from utils.visualisation.misc_functions import save_class_activation_images
    from PIL import Image
    import matplotlib.pyplot as plt
    global save_img_index
    if isinstance(target_layers, list):
        for target_layer in target_layers:
            # Grad cam
            grad_cam = GradCam(model, target_layer=target_layer)  # "transition2.pool")#"denseblock3.denselayer8.relu2")#"conv0")
            # Generate cam mask
            cam = grad_cam.generate_cam(imgs[0].unsqueeze(0), labels[0])
            # original_image
            mean = np.array([0.4914, 0.4822, 0.4465])
            var = np.array([0.2023, 0.1994, 0.2010])  # R,G,B每层的归一化用到的均值和方差
            img = imgs[0].cpu().detach().numpy()
            img = np.transpose(img, (1, 2, 0))
            img = (img * var + mean) * 255  # 去标准化
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            # plt.imshow(img)
            # plt.show()
            # Save mask
            save_class_activation_images(img, cam, "heatmap_" + str(
                save_img_index) + "_GradCAM" + "_L-" + target_layer + "_Label" + str(labels[0].item()))

    # img save
    img.save(savePath + "heatmap_" + str(save_img_index) + "_GradCAM" '_OriImage' + '.png')

    # mask save
    if isinstance(mask, torch.Tensor):
        if mask.shape[0] == 1:
            mask = mask[0].cpu().detach().numpy()
            cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_GradCAM" + '_Mask' + '.png', mask * 255)
        else:
            bar = np.ones((mask.shape[1], 5), dtype=np.float32)
            omask = 0
            l = []
            for i in range(mask.shape[0]):
                l.append(mask[i].cpu().detach().numpy())
                m = cv.hconcat(l)
                l = [m, bar]
                omask = 1 - (1 - omask) * (1 - mask[i].cpu().detach().numpy())
            cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_GradCAM" '_Mask0' + '.png', omask * 255)
            cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_GradCAM" '_Mask1' + '.png', m * 255)

    save_img_index = save_img_index + 1
    print('Grad cam completed')



#CJY MBagNet专属， 显示不同感受野的logit-map
def drawRfLogitsMap(rf_score_maps, num_class, rank_logits_dict, EveryMaxFlag=1, OveralMaxFlag=1, AvgFlag=1):
    global save_img_index
    global savePath

    percentile = 99

    rfs_weight_dict = {}   #linear weight 保存
    label_dict = {}        #label保存
    map_dict = {}          #rf-logits保存
    # 用于分配rf-logit在字典map-dict中的索引
    pos = 0
    row_pos = 0

    #1. 从rf_score_map中依次解压出要展示的东西
    #（1）.提取原图并去标准化
    mean = np.array([0.4914, 0.4822, 0.4465])
    var = np.array([0.2023, 0.1994, 0.2010])  # R,G,B每层的归一化用到的均值和方差

    img = rf_score_maps.pop(0).cpu().detach().numpy()
    img = np.transpose(img, (1,2,0))
    img = img*var+mean   #去标准化

    #img = cv.resize(img, (224,224))
    #img_r = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    r, g, b = cv.split(img)
    img_bgr = cv.merge([b, g, r])
    #cv.imshow("img", img_bgr)
    #cv.waitKey(0)

    #（2）.提取标签：label，p-label （mask-label）
    labels = rf_score_maps.pop(0)
    label = labels[0].cpu().item()
    p_label = labels[1].cpu().item()
    for i in range(num_class+1):
        if i == label+1:
            label_dict[i] = np.ones((1,1))
        elif i == p_label+1:
            label_dict[i] = np.ones((1,1))*(-1)
        else:
            label_dict[i] = np.zeros((1,1))
    if len(labels) == 3:
        mask_label = labels[2].bool().cpu().detach().numpy()
        show_mask_label = 1
    else:
        show_mask_label = 0


    #（3）.显示感受野的权重 or 网络block结构
    rfs_weight_max = 0
    rfs_weight_min = 100000
    rfs_weight = rf_score_maps.pop(0).cpu().detach().numpy()
    for index in range(len(rf_score_maps)-1):
        map = rfs_weight[0][index]
        temp_max = np.max(map)
        temp_min = np.min(map)
        if temp_max > rfs_weight_max:
            rfs_weight_max = temp_max
        if temp_min < rfs_weight_min:
            rfs_weight_min = temp_min

        rfs_weight_dict[index] = map

    if abs(rfs_weight_max) > abs(rfs_weight_min):
        rfs_weight_max = abs(rfs_weight_max)
        rfs_weight_min = -abs(rfs_weight_max)
    else:
        rfs_weight_max = abs(rfs_weight_min)
        rfs_weight_min = -abs(rfs_weight_min)

    #（4）.显示每个感受野的logits-map
    every_rf_logits_absmax = {}   #记录每个感受野下的最大值
    every_rf_logits_mean = {}  #记录每个感受野下的均值
    mean_sum = {}  #记录同类的mean之和
    for index, rf_score in enumerate(rf_score_maps):
        rf_score = rf_score.cpu().detach().numpy()

        # 计算每个rf下abs(logit)的最大值,显示用
        every_rf_logits_absmax[index] = 0
        for i in range(num_class):
            max = np.percentile(np.abs(rf_score[0][i]), percentile)
            if max > every_rf_logits_absmax[index]:
                every_rf_logits_absmax[index] = max

            # 计算每个rf-class-logit-map的均值，用于第二张图显示
            every_rf_logits_mean[pos+row_pos*len(rf_score_maps)] = np.mean(rf_score[0][i], keepdims=True)

            if pos == 0:
                mean_sum[row_pos] = np.mean(rf_score[0][i])
            else:
                mean_sum[row_pos] = mean_sum[row_pos] + np.mean(rf_score[0][i])

            # 将待显示的图按顺序重新排序放入map-dict中，因为plt绘图序号的原因
            map_dict[pos+row_pos*len(rf_score_maps)] = rf_score[0][i]
            row_pos = row_pos + 1

        pos = pos + 1
        row_pos = 0

    #求全局最大的max, 也可以用于显示。至于到底是用every还是overall，可以自己设定
    overall_rf_logits_absmax = 0
    for key in range(len(every_rf_logits_absmax)):
        if every_rf_logits_absmax[key] > overall_rf_logits_absmax:
            overall_rf_logits_absmax = every_rf_logits_absmax[key]


    #2.显示(保存)
    #"""
    #（1）.保存原图
    cv.imwrite(savePath + "heatmap_" + str(save_img_index) + '_0' + '.png', img_bgr * 255)

    #（2）.在padding后的img上绘制拥有最大logits的几个框
    MaxPadding = rank_logits_dict[0][0]["max_padding"]
    color = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 1]]  # 蓝，绿，红，黄，浅蓝，粉，白
    img_padding = cv.copyMakeBorder(img_bgr, MaxPadding, MaxPadding, MaxPadding, MaxPadding, cv.BORDER_CONSTANT,
                                    value=[0, 0, 0])
    for i in range(num_class):
        # r = random.randint(0, 255)/255.0
        # g = random.randint(0, 255)/255.0
        # b = random.randint(0, 255)/255.0
        # color = [b, g, r]
        # print(color)
        if i != label and i != p_label:
            continue
        for rank_logits_inf in rank_logits_dict[i]:
            center_x = rank_logits_inf["center_x"]
            center_y = rank_logits_inf["center_y"]
            half_rf_size = rank_logits_inf["rf_size"] // 2
            LTpoint = (center_x - half_rf_size + MaxPadding, center_y - half_rf_size + MaxPadding)
            RDpoint = (center_x + half_rf_size + MaxPadding, center_y + half_rf_size + MaxPadding)
            cv.rectangle(img_padding, LTpoint, RDpoint, color=color[i], thickness=1, )
            cv.circle(img_padding, (center_x + MaxPadding, center_y + MaxPadding), 1, (0, 0, 255), -1)

    cv.imwrite(savePath + "heatmap_" + str(save_img_index) + '_1' + '.png', img_padding * 255)


    #（3）.绘制rf-logits-map
    # a.每个rf的grad（一列）共用一个最大值
    if EveryMaxFlag == 1:
        num_rf = len(rf_score_maps)
        if show_mask_label == 1:
            window_col = num_rf + 1 + 1  # 最后留一列给mask-label
            window_row = num_class + 1
        else:
            window_col = num_rf + 1
            window_row = num_class + 1

        fig = plt.figure(1, figsize=(window_col, window_row), dpi=120)

        # a-0 显示图片
        ax = plt.subplot(window_row, window_col, 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(img)  # extent=[0,100,0,100],
        plt.axis('off')

        # a-1.显示rfs_weight

        for i in range(len(rfs_weight_dict)):
            # print(i+1+1)
            ax = plt.subplot(window_row, window_col, i + 1 + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(rfs_weight_dict[i], interpolation='none', cmap='RdBu_r', vmin=rfs_weight_min,
                      vmax=rfs_weight_max)  # extent=[0,100,0,100],
            plt.axis('off')


        # a-2.显示label
        for i in range(1, num_class + 1):
            # print(i*(num_rf+1) + 1)
            ax = plt.subplot(window_row, window_col, i * (window_col) + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')

        if show_mask_label == 1:  # 显示masklabel
            mask_combine = np.zeros_like(mask_label[0])
            if mask_label.shape[0] != 1:
                for i in range(mask_label.shape[0]):
                    ax = plt.subplot(window_row, window_col, window_col * (i + 2))
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    ax.imshow(mask_label[i], interpolation='none', cmap='RdBu_r', vmin=-1,
                              vmax=1)  # extent=[0,100,0,100],
                    plt.axis('off')
                    mask_combine = mask_combine | mask_label[i]

                ax = plt.subplot(window_row, window_col, window_col)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                ax.imshow(mask_combine, interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
                plt.axis('off')
            else:
                ax = plt.subplot(window_row, window_col, window_col)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                ax.imshow(mask_label[0], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
                plt.axis('off')

        # a-3.显示rf_score per class
        for i in range(len(map_dict)):
            # print(i+1+(num_rf+1)+i//num_rf+1)
            ax = plt.subplot(window_row, window_col,
                             window_col + 2 + i + (window_col - num_rf) * (i // num_rf))  # i+1+(num_rf+1)+i//num_rf+1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            max = every_rf_logits_absmax[
                i % num_rf]  # np.percentile(np.abs(map_dict[i]), percentile)   #可以调整最大值是overall，还是rf内，还是unit
            # max = overall_rf_logits_absmax
            if max == 0:
                max = 1
            ax.imshow(map_dict[i], interpolation='none', cmap='RdBu_r', vmin=-max, vmax=max)  # extent=[0,100,0,100],
            plt.axis('off')
        # plt.show()

        plt.savefig(savePath + "heatmap_" + str(save_img_index) + '_2' + '.png', bbox_inches='tight', pad_inches=0, dpi=150)


    # b.所有rf的grad共用一个最大值
    if OveralMaxFlag == 1:
        num_rf = len(rf_score_maps)
        if show_mask_label == 1:
            window_col = num_rf + 1 + 1  # 最后留一列给mask-label
            window_row = num_class + 1
        else:
            window_col = num_rf + 1
            window_row = num_class + 1

        fig = plt.figure(1, figsize=(window_col, window_row), dpi=120)

        # b-0 显示图片
        ax = plt.subplot(window_row, window_col, 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(img)  # extent=[0,100,0,100],
        plt.axis('off')

        # b-1.显示rfs_weight
        for i in range(len(rfs_weight_dict)):
            # print(i+1+1)
            ax = plt.subplot(window_row, window_col, i + 1 + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(rfs_weight_dict[i], interpolation='none', cmap='RdBu_r', vmin=rfs_weight_min,
                      vmax=rfs_weight_max)  # extent=[0,100,0,100],
            plt.axis('off')

        # b-2.显示label
        for i in range(1, num_class + 1):
            # print(i*(num_rf+1) + 1)
            ax = plt.subplot(window_row, window_col, i * (window_col) + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')

        if show_mask_label == 1:  # 显示masklabel
            mask_combine = np.zeros_like(mask_label[0])
            if mask_label.shape[0] != 1:
                for i in range(mask_label.shape[0]):
                    ax = plt.subplot(window_row, window_col, window_col * (i + 2))
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    ax.imshow(mask_label[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
                    plt.axis('off')
                    mask_combine = mask_combine | mask_label[i]

                ax = plt.subplot(window_row, window_col, window_col)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                ax.imshow(mask_combine, interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
                plt.axis('off')
            else:
                ax = plt.subplot(window_row, window_col, window_col)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                ax.imshow(mask_label[0], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
                plt.axis('off')

        # b-3.显示rf_score per class
        for i in range(len(map_dict)):
            # print(i+1+(num_rf+1)+i//num_rf+1)
            ax = plt.subplot(window_row, window_col, window_col + 2 + i + (window_col - num_rf) * (i // num_rf))  # i+1+(num_rf+1)+i//num_rf+1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            #max = every_rf_logits_absmax[i % num_rf]  # np.percentile(np.abs(map_dict[i]), percentile)   #可以调整最大值是overall，还是rf内，还是unit
            max = overall_rf_logits_absmax
            if max == 0:
                max = 1
            ax.imshow(map_dict[i], interpolation='none', cmap='RdBu_r', vmin=-max, vmax=max)  # extent=[0,100,0,100],
            plt.axis('off')
        # plt.show()

        plt.savefig(savePath + "heatmap_" + str(save_img_index) + '_3' + '.png', bbox_inches='tight', pad_inches=0, dpi=150)

    # c.显示每个grad的mean, 所有rf的grad共用一个最大值
    if AvgFlag == 1:
        fig = plt.figure(1, figsize=(num_rf + 1, num_class + 1), dpi=120)
        ax = plt.subplot(num_class + 1, num_rf + 1, 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(img)  # extent=[0,100,0,100],
        plt.axis('off')

        # c-1.显示rfs_weight
        for i in range(len(rfs_weight_dict)):
            # print(i+1+1)
            ax = plt.subplot(num_class + 1, num_rf + 1, i + 1 + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(rfs_weight_dict[i], interpolation='none', cmap='RdBu_r', vmin=rfs_weight_min,
                      vmax=rfs_weight_max)  # extent=[0,100,0,100],
            plt.axis('off')

        # c-2.显示label
        for i in range(1, num_class + 1):
            # print(i*(num_rf+1) + 1)
            ax = plt.subplot(num_class + 1, num_rf + 1, i * (num_rf + 1) + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')


        # c-3.显示rf_score per class
        mean_max = 0
        for i in range(len(every_rf_logits_mean)):
            if abs(every_rf_logits_mean[i][0][0])>mean_max:
                mean_max = abs(every_rf_logits_mean[i][0][0])

        for i in range(len(map_dict)):
            # print(i+1+(num_rf+1)+i//num_rf+1)
            ax = plt.subplot(num_class + 1, num_rf + 1, i + 1 + (num_rf + 1) + i // num_rf + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            #max = every_rf_logits_absmax[i%num_rf]#np.percentile(np.abs(map_dict[i]), percentile)   #可以调整最大值是overall，还是rf内，还是unit
            ax.imshow(every_rf_logits_mean[i], interpolation='none', cmap='RdBu_r', vmin=-mean_max, vmax=mean_max)  # extent=[0,100,0,100],
            #print(every_rf_logits_mean[i])
            plt.axis('off')
        #plt.show()

        plt.savefig(savePath + "heatmap_" + str(save_img_index) + '_4' + '.png', bbox_inches='tight', pad_inches=0, dpi=150)

    # CJY at 2020.2.27
    if show_mask_label == 1:  # 显示masklabel
        w = num_class + 1
        h = 3
    else:
        w = num_class + 1
        h = 2

    fig = plt.figure(2, figsize=(w, h), )  # dpi=120)

    # 显示标签
    for i in range(1, num_class + 1):
        # print(i*(num_rf+1) + 1)
        ax = plt.subplot(h, w, i+1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
        plt.axis('off')

    # 显示图片
    ax = plt.subplot(h, w, w + 1)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    ax.imshow(img)  # extent=[0,100,0,100],
    plt.axis('off')

    max = every_rf_logits_absmax[len(every_rf_logits_absmax)-1]
    seg = rf_score_maps[-1].cpu().detach().numpy()
    for i in range(num_class):
        ax = plt.subplot(h, w, w + i + 2)
        ax.imshow(seg[0][i], interpolation='none', cmap='RdBu_r', vmin=-max, vmax=max)
        plt.axis('off')


    if show_mask_label == 1:  # 显示masklabel
        mask_combine = np.zeros_like(mask_label[0])
        if mask_label.shape[0] != 1:
            for i in range(mask_label.shape[0]):
                ax = plt.subplot(h, w, 2*w + i + 2)
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                ax.imshow(mask_label[i], interpolation='none', cmap='RdBu_r', vmin=-1,
                          vmax=1)  # extent=[0,100,0,100],
                plt.axis('off')
                mask_combine = mask_combine | mask_label[i]

            ax = plt.subplot(h, w, 2*w+1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(mask_combine, interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')
        else:
            ax = plt.subplot(h, w, 2*w+1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(mask_label[0], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')

    plt.savefig(savePath + "heatmap_" + str(save_img_index) + '_5' + '.png', bbox_inches='tight', pad_inches=0,
                    dpi=150)



    # 记录保存图片的索引的全局变量
    save_img_index = save_img_index + 1


def drawDenseFCMask(img, seg, mask=None):
    global savePath
    global save_img_index
    # 1.提取原图并去标准化,保存
    mean = np.array([0.4914, 0.4822, 0.4465])
    var = np.array([0.2023, 0.1994, 0.2010])  # R,G,B每层的归一化用到的均值和方差

    img = img.cpu().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * var + mean  # 去标准化

    # img = cv.resize(img, (224,224))
    # img_r = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    r, g, b = cv.split(img)
    img_bgr = cv.merge([b, g, r])
    # cv.imshow("img", img_bgr)
    # cv.waitKey(0)
    cv.imwrite(savePath + "heatmap_" +str(save_img_index) + "_denseFC" + '_img' + '.png', img_bgr * 255)

    # 2. seg
    if seg.shape[0] == 1:
        seg = seg[0].cpu().detach().numpy()
        cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_denseFC" '_seg' +'.png', seg * 255)
    else:
        bar = np.ones((seg.shape[1], 5), dtype=np.float32)
        l = []
        for i in range(seg.shape[0]):
            l.append(seg[i].cpu().detach().numpy())
            s = cv.hconcat(l)
            l = [s, bar]
        cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_denseFC" '_seg' + '.png', s * 255)

    # 3. mask
    if isinstance(mask, torch.Tensor):
        if mask.shape[0] == 1:
            mask = mask[0].cpu().detach().numpy()
            cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_denseFC" + '_mask' + '.png', mask * 255)
        else:
            bar = np.ones((mask.shape[1], 5), dtype=np.float32)
            l = []
            for i in range(mask.shape[0]):
                l.append(mask[i].cpu().detach().numpy())
                m = cv.hconcat(l)
                l = [m, bar]
            cv.imwrite(savePath + "heatmap_" + str(save_img_index) + "_denseFC" '_mask' + '.png', m * 255)


    # 记录保存图片的索引的全局变量
    save_img_index = save_img_index + 1


