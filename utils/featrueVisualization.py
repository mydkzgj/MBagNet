
#做一个可视化
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

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


#CJY 显示不同感受野的特征 版本1  最终采用非线性分类器
def showrfFeatureMap(rf_score_maps, num_class, rank_logits_dict, AvgFlag=1):
    size = (60, 60)
    percentile = 99

    rfs_weight_dict = {}
    label_dict = {}
    map_dict = {}
    pos = 0
    row_pos = 0

    #1.将原图显示
    mean = np.array([0.4914, 0.4822, 0.4465])
    var = np.array([0.2023, 0.1994, 0.2010])  # R,G,B每层的归一化用到的均值和方差

    img = rf_score_maps.pop(0).cpu().detach().numpy()
    img = np.transpose(img, (1,2,0))
    img = img*var+mean   #去标准化

    #img = cv.resize(img, (224,224))
    #img_r = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    r, g, b = cv.split(img)
    img_bgr = cv.merge([b, g, r])
    cv.imshow("img", img_bgr)
    cv.waitKey(0)

    # 将最大logits的几个框显示出来
    MaxPadding = rank_logits_dict[0][0]["max_padding"]
    color = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,1,0],[1,0,1],[1,1,1]]   #蓝，绿，红，黄，浅蓝，粉，白
    img_padding = cv.copyMakeBorder(img_bgr, MaxPadding, MaxPadding, MaxPadding, MaxPadding, cv.BORDER_CONSTANT,value=[0,0,0])
    for i in range(num_class):
        #r = random.randint(0, 255)/255.0
        #g = random.randint(0, 255)/255.0
        #b = random.randint(0, 255)/255.0
        #color = [b, g, r]
        #print(color)
        if i!=3:
            continue
        for rank_logits_inf in rank_logits_dict[i]:
            center_x = rank_logits_inf["center_x"]
            center_y = rank_logits_inf["center_y"]
            half_rf_size = rank_logits_inf["rf_size"]//2
            LTpoint = (center_x-half_rf_size+MaxPadding, center_y-half_rf_size+MaxPadding)
            RDpoint = (center_x+half_rf_size+MaxPadding, center_y+half_rf_size+MaxPadding)
            cv.rectangle(img_padding, LTpoint, RDpoint, color=color[i], thickness=1,)

    cv.imshow("imgP", img_padding)
    cv.waitKey(0)

    #2.将标签显示
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

    #3.显示感受野的权重
    rfs_weight_max = 0
    rfs_weight_min = 100000
    rfs_weight = rf_score_maps.pop(0).cpu().detach().numpy()
    for index in range(len(rf_score_maps)):
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

    #4.显示每个感受野不同类的关注点
    max_rf = {}  #记录每个感受野下的最大值
    mean_rf_dict = {}  #记录每个感受野下的最小值
    mean_sum = {}  #记录同类的mean之和
    for index, rf_score in enumerate(rf_score_maps):
        rf_score = rf_score.cpu().detach().numpy()
        num_batch = rf_score.shape[0]
        num_classes = rf_score.shape[1]

        max_rf[index] = 0


        for i in range(num_classes):
            max = np.percentile(np.abs(rf_score[0][i]), percentile)

            if max > max_rf[index]:
                max_rf[index] = max

            """ #for opencv
            map = rf_score[0][i]/(max+1e-12)   #for opencv
            map = cv.resize(map, size)
            if i == 0:
                vmap = map
            else:
                vmap = cv.vconcat([vmap, map])
            """

            mean_rf_dict[pos+row_pos*len(rf_score_maps)] = np.mean(rf_score[0][i],keepdims=True)
            if pos == 0:
                mean_sum[row_pos] = np.mean(rf_score[0][i])
            else:
                mean_sum[row_pos] = mean_sum[row_pos] + np.mean(rf_score[0][i])

            map_dict[pos+row_pos*len(rf_score_maps)] = rf_score[0][i]
            row_pos = row_pos + 1

        """
        if index == 0:
            hmap = vmap
        else:
            hmap = cv.hconcat([hmap, vmap])
        """
        pos = pos + 1
        row_pos = 0

    #hmap = cv.vconcat([rfs_hmap, hmap])
    #cv.imshow("feature",hmap)
    #cv.waitKey(0)

    # 如果求全局最大呢？
    overall_max = 0
    for key in max_rf.keys():
        if max_rf[key]>overall_max:
            overall_max = max_rf[key]


    #5.显示
    #"""
    num_rf = len(rf_score_maps)

    fig1 = plt.figure(1,figsize=(num_rf+1, num_classes+1), dpi=120)

    #5-0 显示图片
    ax = plt.subplot(num_classes + 1, num_rf + 1, 1)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    ax.imshow(img)  # extent=[0,100,0,100],
    plt.axis('off')

    #5-1.显示rfs_weight
    for i in range(len(rfs_weight_dict)):
        #print(i+1+1)
        ax = plt.subplot(num_classes+1, num_rf+1, i + 1 + 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(rfs_weight_dict[i], interpolation='none', cmap='RdBu_r', vmin=rfs_weight_min, vmax=rfs_weight_max)  # extent=[0,100,0,100],
        plt.axis('off')

    #5-2.显示label
    for i in range(1,num_classes+1):
        #print(i*(num_rf+1) + 1)
        ax = plt.subplot(num_classes+1, num_rf+1, i*(num_rf+1) + 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
        plt.axis('off')

    #5-3.显示rf_score per class
    for i in range(len(map_dict)):
        #print(i+1+(num_rf+1)+i//num_rf+1)
        ax = plt.subplot(num_classes+1, num_rf+1, i+1+(num_rf+1)+i//num_rf+1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #max = max_rf[i%num_rf]#np.percentile(np.abs(map_dict[i]), percentile)   #可以调整最大值是overall，还是rf内，还是unit
        max = overall_max
        if max == 0:
            max = 1
        ax.imshow(map_dict[i], interpolation='none', cmap='RdBu_r', vmin=-max, vmax=max)  #extent=[0,100,0,100],
        plt.axis('off')
    #plt.show()

    if AvgFlag == 1:
        fig2 = plt.figure(2, figsize=(num_rf + 1, num_classes + 1), dpi=120)
        ax = plt.subplot(num_classes + 1, num_rf + 1, 1)
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        ax.imshow(img)  # extent=[0,100,0,100],
        plt.axis('off')

        # 5-1.显示rfs_weight
        for i in range(len(rfs_weight_dict)):
            # print(i+1+1)
            ax = plt.subplot(num_classes + 1, num_rf + 1, i + 1 + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(rfs_weight_dict[i], interpolation='none', cmap='RdBu_r', vmin=rfs_weight_min,
                      vmax=rfs_weight_max)  # extent=[0,100,0,100],
            plt.axis('off')

        # 5-2.显示label
        for i in range(1, num_classes + 1):
            # print(i*(num_rf+1) + 1)
            ax = plt.subplot(num_classes + 1, num_rf + 1, i * (num_rf + 1) + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ax.imshow(label_dict[i], interpolation='none', cmap='RdBu_r', vmin=-1, vmax=1)  # extent=[0,100,0,100],
            plt.axis('off')

        # 5-3.显示rf_score per class
        mean_max = 0
        for i in range(len(mean_rf_dict)):
            if abs(mean_rf_dict[i][0][0])>mean_max:
                mean_max = abs(mean_rf_dict[i][0][0])

        for i in range(len(map_dict)):
            # print(i+1+(num_rf+1)+i//num_rf+1)
            ax = plt.subplot(num_classes + 1, num_rf + 1, i + 1 + (num_rf + 1) + i // num_rf + 1)
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            # max = max_rf[i%num_rf]#np.percentile(np.abs(map_dict[i]), percentile)   #可以调整最大值是overall，还是rf内，还是unit
            ax.imshow(mean_rf_dict[i], interpolation='none', cmap='RdBu_r', vmin=-mean_max, vmax=mean_max)  # extent=[0,100,0,100],
            #print(mean_rf_dict[i])
            plt.axis('off')
        plt.show()

        # 为了验证是否显示准确
        """
        for i in range(len(map_dict)):
            if i%23 == 0:
                sum = 0
            else:
                sum = sum + mean_rf_dict[i]
            if i%23 == 22:
                print(sum)
        """
    #"""



#CJY 显示不同感受野的特征 版本1  最终采用非线性分类器
#def showRankBBox(rf_score_maps, num_class, AvgFlag=1):