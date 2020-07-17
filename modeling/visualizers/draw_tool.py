"""
Created on 2020.7.4

@author: Jiayang Chen - github.com/mydkzgj
"""


import os
import cv2 as cv
import numpy as np

# 伪彩色图映射方式
def colormap_name(id):
    switcher = {
        0 : "COLORMAP_AUTUMN",
        1 : "COLORMAP_BONE",
        2 : "COLORMAP_JET",
        3 : "COLORMAP_WINTER",
        4 : "COLORMAP_RAINBOW",
        5 : "COLORMAP_OCEAN",
        6 : "COLORMAP_SUMMER",
        7 : "COLORMAP_SPRING",
        8 : "COLORMAP_COOL",
        9 : "COLORMAP_HSV",
        10: "COLORMAP_PINK",
        11: "COLORMAP_HOT"
    }
    return switcher.get(id, 'NONE')

# 色彩映射  注：bgr
def color_name(id):
    switcher = {
        0 : [0,255,0], #紫[255,48,155],
        1 : [0,255,255],
        2 : [0,165,255],  #255 165 0   #224,255,255
        3 : [0,0,255],
        4 : [0,255,255],
        5 : [255,0,255],
        6 : [255,255,255],
        7 : "COLORMAP_SPRING",
        8 : "COLORMAP_COOL",
        9 : "COLORMAP_HSV",
        10: "COLORMAP_PINK",
        11: "COLORMAP_HOT"
    }
    return switcher.get(id, 'NONE')


def draw_visualization(img, visualization, gtmask, binary_threshold, savePath, index_prefix, label_prefix, visual_prefix):
    """
    single img
    """
    # 1.将Tensor转化为numpy
    img_numpy, visual_numpy, gtmask_numpy = convertTensorToNumpy(img, visualization, gtmask)

    # 2.将visualization_numpy转化为伪彩色图
    if len(visual_numpy.shape) == 2:
        visual_numpy_color = cv.applyColorMap(visual_numpy, cv.COLORMAP_JET)
    else:  # for 3-channels visualization  (Guided Backpropagation)  [0,255]
        visual_numpy_color = visual_numpy
        # 灰度图的合成方式有两种：1.abs sum  2. non-zero binary
        visual_numpy = np.mean(np.abs((visual_numpy_color-127.0))*2, axis=2)#cv.cvtColor(visual_numpy_color, cv.COLOR_BGR2GRAY)
        visual_numpy = (visual_numpy/visual_numpy.max() *255).astype(np.uint8)
        #th, visual_numpy = cv.threshold(np.mean(np.abs(visual_numpy_color-127.0), axis=2).astype(np.uint8), 0, 255, cv.THRESH_BINARY)
    th, visual_numpy_binary = cv.threshold(visual_numpy, int(binary_threshold*255), 255, cv.THRESH_BINARY)

    # 3.将visualization和img叠加
    img_ratio = 0.8
    visual_ratio = 1 - img_ratio
    img_with_visual = cv.addWeighted(img_numpy, img_ratio, visual_numpy_color, visual_ratio, 0)

    visual_numpy_binary = cv.cvtColor(visual_numpy_binary, cv.COLOR_GRAY2BGR)
    gtmask_with_visual = cv.addWeighted(gtmask_numpy, 0.5, visual_numpy_binary, 0.5, 0)

    # 4.Save
    if index_prefix != "":
        index_prefix = index_prefix + "_"
    cv.imwrite(os.path.join(savePath, "{}image_{}.jpg".format(index_prefix, label_prefix)), img_numpy)
    cv.imwrite(os.path.join(savePath, "{}visualization_gray_{}_{}.jpg".format(index_prefix, visual_prefix, label_prefix)), visual_numpy)
    #cv.imwrite(os.path.join(savePath, "{}visualization_binary_{}_{}_th{}.jpg".format(index_prefix, visual_prefix, label_prefix, str(binary_threshold))), visual_numpy_binary)
    cv.imwrite(os.path.join(savePath, "{}visualization_color_{}_{}.jpg".format(index_prefix, visual_prefix, label_prefix)), visual_numpy_color)
    #cv.imwrite(os.path.join(savePath, "{}visualization_on_image_{}_{}.jpg".format(index_prefix, visual_prefix, label_prefix)),img_with_visual)
    cv.imwrite(os.path.join(savePath, "{}visualization_binary_on_gtmask_{}_{}_th{}.jpg".format(index_prefix, visual_prefix, label_prefix, str(binary_threshold))), gtmask_with_visual)
    cv.imwrite(os.path.join(savePath, "{}segmentation_ground_truth_{}.jpg".format(index_prefix, label_prefix)), gtmask_numpy)
    if gtmask_numpy.shape[2] == 3:
        gt_gray = cv.cvtColor(gtmask_numpy, cv.COLOR_RGB2GRAY)
        th, gt_binary = cv.threshold(gt_gray, 0, 255, cv.THRESH_BINARY)
        #cv.imwrite(os.path.join(savePath, "{}segmentation_ground_truth_binary_{}.jpg".format(index_prefix, label_prefix)), gt_binary)



def convertTensorToNumpy(img, visualization, gtmask):
    # (1). 转化原图
    mean = np.array([0.4914, 0.4822, 0.4465])
    var = np.array([0.2023, 0.1994, 0.2010])  # R,G,B每层的归一化用到的均值和方差
    # 图像tensor与numpy的通道顺序不同
    img_numpy = img.cpu().detach().numpy()
    img_numpy = np.transpose(img_numpy, (1,2,0))
    img_numpy = img_numpy * var + mean    #去标准化
    # 图像Resize
    #img = cv.resize(img, (224,224))
    #img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    r, g, b = cv.split(img_numpy)
    img_numpy_bgr = cv.merge([b, g, r])
    #cv.imshow("img", img_bgr)
    #cv.waitKey(0)
    img_numpy = (img_numpy_bgr * 255).astype(np.uint8)

    # (2).转化可视化化结果 （单通道）
    visual_numpy = visualization.permute(1,2,0).squeeze(2).cpu().detach().numpy()
    if visual_numpy.shape[1] != img_numpy.shape[1]:
        visual_numpy = cv.resize(visual_numpy, (img_numpy.shape[0], img_numpy.shape[1]), interpolation=cv.INTER_LINEAR)   #https://blog.csdn.net/xidaoliang/article/details/86504720
    #visual_numpy = (128 + visual_numpy * 127).astype(np.uint8)
    visual_numpy = (visual_numpy * 255).astype(np.uint8)

    # (3).转化Ground Truth (多/单通道) 转化为 彩图/灰度图
    if gtmask is not None:
        gtmask_channel = gtmask.shape[0]
        if gtmask_channel == 1:
            gtmask_numpy = gtmask.cpu().detach().numpy()
        else:
            mutichannel_gtmask_numpy = gtmask.cpu().detach().numpy()
            color_gtmask_numpy = 0
            for i in range(gtmask_channel):
                single_gtmask_numpy = mutichannel_gtmask_numpy[i]
                single_gtmask_numpy = np.expand_dims(single_gtmask_numpy, axis=2).repeat(3, axis=2)
                color_gtmask_numpy = color_gtmask_numpy + single_gtmask_numpy * np.array(color_name(i))
            gtmask_numpy = color_gtmask_numpy.astype(np.uint8)
    else:
        gtmask_numpy = img_numpy

    return img_numpy, visual_numpy, gtmask_numpy



