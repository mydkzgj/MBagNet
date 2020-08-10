# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import os

import torch
import torch.nn as nn
import torchvision.transforms as tran
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from ignite.metrics import ConfusionMatrix

from ignite.contrib.metrics import ROC_AUC

from sklearn.metrics import roc_curve, auc
from utils.plot_ROC import plotROC_OneClass, plotROC_MultiClass
from utils.draw_ConfusionMatrix import drawConfusionMatrix
import numpy as np
import pandas as pd

import utils.featrueVisualization as fv

import random
import copy

import cv2 as cv

"""
# pytorch 转换 one-hot 方式 scatter
def activated_output_transform(output):
    y_pred = output["logits"]
    y_pred = torch.sigmoid(y_pred)
    labels = output["labels"]
    labels_one_hot = torch.FloatTensor(y_pred.shape[0], y_pred.shape[1])
    labels_one_hot.scatter_(1, labels.cpu().unsqueeze(1), 1).cuda()
    return y_pred, labels_one_hot
"""

imgsName = []
segmentationMetric = {}  # 用于保存TP,FP,TN,FN

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

# 计算逐像素的tp，tn等
def prepareForComputeSegMetric(seg_map, seg_mask, label, layer_name, th=0.5):   # 适用于多标签
    if seg_map.shape[1] == 3:  #如果输入是3通道的visualization （如Guided Backpropagation），转为单通道
        seg_map = torch.mean((seg_map-0.5).abs()*2, dim=1, keepdim=True)
        seg_map = seg_map/seg_map.max()

    seg_pmask = torch.gt(seg_map, th)
    if seg_pmask.shape[1] == 1:
        seg_pmask = seg_pmask.expand_as(seg_mask)
    seg_mask = seg_mask.bool()

    tp = (seg_pmask & seg_mask).sum(-1).sum(-1)
    fp = (seg_pmask & (~seg_mask)).sum(-1).sum(-1)
    tn = ((~seg_pmask) & (~seg_mask)).sum(-1).sum(-1)
    fn = ((~seg_pmask) & seg_mask).sum(-1).sum(-1)

    for i in range(seg_mask.shape[0]):
        label_name = label[i].item()
        global segmentationMetric
        if segmentationMetric.get(layer_name) == None:
            segmentationMetric[layer_name] = {}

        if segmentationMetric[layer_name].get(th) == None:
            segmentationMetric[layer_name][th] = {}

        if segmentationMetric[layer_name][th].get(label_name) == None:
            segmentationMetric[layer_name][th][label_name] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

        segmentationMetric[layer_name][th][label_name]["TP"] = segmentationMetric[layer_name][th][label_name]["TP"] + tp[i]
        segmentationMetric[layer_name][th][label_name]["FP"] = segmentationMetric[layer_name][th][label_name]["FP"] + fp[i]
        segmentationMetric[layer_name][th][label_name]["TN"] = segmentationMetric[layer_name][th][label_name]["TN"] + tn[i]
        segmentationMetric[layer_name][th][label_name]["FN"] = segmentationMetric[layer_name][th][label_name]["FN"] + fn[i]

    return tp, fp, tn, fn


def fillHoles(seedImg, Mask):
    # 输入是numpy格式
    seedImg = seedImg * Mask
    #cv.imshow("1", seedImg[:, :, -1]*255)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    i = 0
    while 1:
        dilated = cv.dilate(seedImg, kernel)  # 膨胀图像
        outputImg = dilated & Mask
        i = i + 1
        if i % 30 == 0:
            #print(i)
            if (seedImg == outputImg).all():
                break
            else:
                seedImg = outputImg
        else:
            seedImg = outputImg

    #cv.imshow("2", outputImg[:, :, -1]*255)
    #cv.imshow("3", Mask[:, :, -1] * 255)
    #cv.imshow("4", Mask[:, :, -1]*(1-outputImg[:, :, -1]) * 255)
    #cv.waitKey(0)
    return outputImg

# 计算逐病灶的tp，tn等（击中）
def prepareForComputeSegMetric2(seg_map, seg_mask, label, layer_name, th=0.5):   # 适用于多标签
    if seg_map.shape[1] == 3:  #如果输入是3通道的visualization （如Guided Backpropagation），转为单通道
        seg_map = torch.mean((seg_map-0.5).abs()*2, dim=1, keepdim=True)
        seg_map = seg_map/seg_map.max()

    seg_pmask = torch.gt(seg_map, th)
    if seg_pmask.shape[1] == 1:
        seg_pmask = seg_pmask.expand_as(seg_mask)
    seg_mask = seg_mask.bool()

    # 计算object-level 和 pixel-level metrics
    hit = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    miss = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    wrong = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    prediction = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    groundtruth = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)

    pixel_tp = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    pixel_fp = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    pixel_tn = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)
    pixel_fn = np.zeros((seg_mask.shape[0], seg_mask.shape[1]), dtype=np.int64)

    for i in range(seg_mask.shape[0]):  #numpy (w, h, channel)
        # 1.计算object-level的参数   hit + wrong =？ predict   hit + miss = groundthruth
        # 注：hit + wrong 不一定等于predict，但后面还是以hit+wrong做分母
        pt_mask = seg_pmask[i].permute(1, 2, 0).numpy().astype(np.uint8)
        gt_mask = seg_mask[i].permute(1, 2, 0).numpy().astype(np.uint8)

        hit_mask = fillHoles(pt_mask, gt_mask)     #保留的是二者相交的gt那部分
        miss_mask = gt_mask - hit_mask
        right_mask = fillHoles(gt_mask, pt_mask)   #保留的是二者相交的pt那部分
        wrong_mask = pt_mask - right_mask

        """
        cv.imshow("1", pt_mask[:, :, -1] * 255)
        cv.imshow("2", gt_mask[:, :, -1] * 255)
        cv.imshow("3", hit_mask[:, :, -1] * 255)
        cv.imshow("4", miss_mask[:, :, -1] * 255)
        cv.imshow("5", right_mask[:, :, -1] * 255)
        cv.imshow("6", wrong_mask[:, :, -1] * 255)
        cv.waitKey(0)
        #"""

        for j in range(seg_mask.shape[1]):
            #pt_retval, pt_labels, pt_stats, pt_centroids = cv.connectedComponentsWithStats(pt_mask[:, :, j], connectivity=8, ltype=cv.CV_32S)
            #prediction[i][j] = pt_retval - 1
            #gt_retval, gt_labels, gt_stats, gt_centroids = cv.connectedComponentsWithStats(gt_mask[:, :, j], connectivity=8, ltype=cv.CV_32S)
            #groundtruth[i][j] = gt_retval - 1
            hit_retval, hit_labels, hit_stats, hit_centroids = cv.connectedComponentsWithStats(hit_mask[:, :, j], connectivity=8, ltype=cv.CV_32S)
            hit[i][j] = hit_retval-1
            miss_retval, miss_labels, miss_stats, miss_centroids = cv.connectedComponentsWithStats(miss_mask[:, :, j], connectivity=8, ltype=cv.CV_32S)
            miss[i][j] = miss_retval - 1
            wrong_retval, wrong_labels, wrong_stats, wrong_centroids = cv.connectedComponentsWithStats(wrong_mask[:, :, j], connectivity=8, ltype=cv.CV_32S)
            wrong[i][j] = wrong_retval - 1

        # 2.计算pixel-level的参数   考虑相交的 right_mask(pt) 和 hit_mask(gt)
        pixel_tp[i] = (right_mask * hit_mask).sum(axis=(0,1))
        pixel_fp[i] = (right_mask * (1-hit_mask)).sum(axis=(0,1))
        pixel_tn[i] = ((1-right_mask) * (1-hit_mask)).sum(axis=(0,1))
        pixel_fn[i] = ((1-right_mask) & hit_mask).sum(axis=(0,1))

    # 按图片计算metric
    hit = torch.Tensor(hit)
    miss = torch.Tensor(miss)
    wrong = torch.Tensor(wrong)

    object_precision = hit/(hit+wrong).clamp(min=1E-12)
    object_recall = hit/(hit+miss).clamp(min=1E-12)

    pixel_tp = torch.Tensor(pixel_tp)
    pixel_fp = torch.Tensor(pixel_fp)
    pixel_tn = torch.Tensor(pixel_tn)
    pixel_fn = torch.Tensor(pixel_fn)

    # 记录在全局变量中
    for i in range(seg_mask.shape[0]):
        label_name = label[i].item()

        global segmentationMetric
        if segmentationMetric.get(layer_name) == None:
            segmentationMetric[layer_name] = {}

        if segmentationMetric[layer_name].get(th) == None:
            segmentationMetric[layer_name][th] = {}

        if segmentationMetric[layer_name][th].get(label_name) == None:
            segmentationMetric[layer_name][th][label_name] = {"HIT":0, "MISS":0, "WRONG":0, "TP": 0, "FP": 0, "TN": 0, "FN": 0}

        segmentationMetric[layer_name][th][label_name]["HIT"] = segmentationMetric[layer_name][th][label_name]["HIT"] + hit[i]
        segmentationMetric[layer_name][th][label_name]["MISS"] = segmentationMetric[layer_name][th][label_name]["MISS"] + miss[i]
        segmentationMetric[layer_name][th][label_name]["WRONG"] = segmentationMetric[layer_name][th][label_name]["WRONG"] + wrong[i]
        segmentationMetric[layer_name][th][label_name]["TP"] = segmentationMetric[layer_name][th][label_name]["TP"] + pixel_tp[i]
        segmentationMetric[layer_name][th][label_name]["FP"] = segmentationMetric[layer_name][th][label_name]["FP"] + pixel_fp[i]
        segmentationMetric[layer_name][th][label_name]["TN"] = segmentationMetric[layer_name][th][label_name]["TN"] + pixel_tn[i]
        segmentationMetric[layer_name][th][label_name]["FN"] = segmentationMetric[layer_name][th][label_name]["FN"] + pixel_fn[i]

    return 0


"""
# 用于统计分割样本集中 病变等级与病灶之间的相关性
LS = {}
def lesionsStatistics(mask, label):
    avgmask = torch.nn.functional.adaptive_max_pool2d(mask, 1)
    avgmask = avgmask.squeeze(-1).squeeze(-1).squeeze(0)
    index = (avgmask[0]*8 + avgmask[1]*4 + avgmask[2]*2 + avgmask[3]).item()
    global LS
    if LS.get(label) == None:
        LS[label] = {}

    if LS[label].get(index) == None:
        LS[label][index] = 0

    LS[label][index] = LS[label][index] + 1
"""




def create_supervised_visualizer(model, metrics, loss_fn, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        grade_imgs, grade_labels, seg_imgs, seg_masks, seg_labels, gimg_path, simg_path = batch
        grade_imgs = grade_imgs.to(device) if torch.cuda.device_count() >= 1 and grade_imgs is not None else grade_imgs
        grade_labels = grade_labels.to(device) if torch.cuda.device_count() >= 1 and grade_labels is not None  else grade_labels
        seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 and seg_imgs is not None else seg_imgs
        seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 and seg_masks is not None else seg_masks
        seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 and seg_labels is not None else seg_labels

        model.transmitClassifierWeight()  # 该函数是将baseline中的finalClassifier的weight传回给base，使得其可以直接计算logits-map，
        model.transimitBatchDistribution(1)  # 所有样本均要生成可视化seg

        dataType = "grade"
        heatmapType = "visualization"  # "GradCAM"#"segmenters"#"GradCAM"#"computeSegMetric"  # "grade", "segmenters", "computeSegMetric", "GradCAM"
        savePath = r"D:\MIP\Experiment\1"  #r"D:\graduateStudent\eyes datasets\cjy\visualization"#

        # grade_labels  #242 boxer, 243 bull mastiff p, 281 tabby cat p,282 tiger cat, 250 Siberian husky, 333 hamster
        if dataType == "grade":
            imgs = grade_imgs
            labels = torch.zeros_like(grade_labels) + 1#333#243
            masks = None
            img_paths = gimg_path
        elif dataType == "seg":
            imgs = seg_imgs
            labels = seg_labels
            masks = seg_masks
            img_paths = simg_path
        elif dataType == "joint":
            imgs = torch.cat([grade_imgs, seg_imgs], dim=0)
            labels = torch.cat([grade_labels, seg_labels], dim=0)
            masks = seg_masks
            img_paths = gimg_path + simg_path

        if heatmapType == "segmentation":
            with torch.no_grad():
                logits = model(imgs)
                scores = torch.softmax(logits, dim=1)
                p_labels = torch.argmax(logits, dim=1)  # predict_label
                return {"logits": logits, "labels": labels}

        elif heatmapType == "visualization":
            # 由于需要用到梯度，所以就不加入with torch.no_grad()了
            logits = model(imgs)
            scores = torch.softmax(logits, dim=1)
            p_labels = torch.argmax(logits, dim=1)  # predict_label

            #"""
            global imgsName
            if imgsName == []:
                imgsName = ["{}".format(i) for i in range(imgs.shape[0])]
            else:
                imgsName = [str(int(i)+imgs.shape[0]) for i in imgsName]
            #"""
            #imgsName = [os.path.split(img_path)[1].split(".")[0] for img_path in img_paths]

            oblabelList = [labels]
            #oblabelList = [p_labels]
            #oblabelList = [labels, p_labels]
            #oblabelList = [labels*0 + i for i in range(model.num_classes)]
            oblabelList = [labels*243, labels*250, labels*281, labels*333]

            # 将读取的数据名字记录下来
            for oblabels in oblabelList:
                binary_threshold = 0.25#0.5
                showFlag = 1
                input_size = (imgs.shape[2], imgs.shape[3])
                visual_num = imgs.shape[0]
                gcam_list, gcam_max_list, overall_gcam = model.visualizer.GenerateVisualiztions(logits, oblabels, input_size, visual_num=visual_num)

                # 用于visualizaztion的数据
                vimgs = imgs[imgs.shape[0] - visual_num:imgs.shape[0]]
                vlabels = labels[imgs.shape[0] - visual_num:imgs.shape[0]]
                vplabels = p_labels[imgs.shape[0] - visual_num:imgs.shape[0]]
                vmasks = masks[masks.shape[0] - visual_num:masks.shape[0]] if masks is not None else None

                if showFlag == 1:
                    # 绘制可视化结果
                    model.visualizer.DrawVisualization(vimgs, vlabels, vplabels, vmasks, binary_threshold, savePath, imgsName)

                if dataType == "seg":
                    # 计算Metric
                    binary_gtmasks = torch.max(vmasks, dim=1, keepdim=True)[0]
                    gtmasks = torch.cat([vmasks, 1 - binary_gtmasks, binary_gtmasks], dim=1)
                    for i, v in enumerate(gcam_list):
                        rv = torch.nn.functional.interpolate(v, input_size, mode='bilinear')
                        segmentations = rv  # .gt(binary_threshold)
                        prepareForComputeSegMetric2(segmentations.cpu(), gtmasks.cpu(),
                                                   labels[imgs.shape[0] - visual_num:imgs.shape[0]],
                                                   layer_name=model.visualizer.target_layer[i], th=binary_threshold)

                    rv = torch.nn.functional.interpolate(overall_gcam, input_size, mode='bilinear')
                    segmentations = rv  # .gt(binary_threshold)
                    prepareForComputeSegMetric2(segmentations.cpu(), gtmasks.cpu(),
                                               labels[imgs.shape[0] - visual_num:imgs.shape[0]],
                                               layer_name="overall", th=binary_threshold)

            return {"logits": logits.detach(), "labels": labels, }



        elif model.heatmapType == "computeSegMetric":
            with torch.no_grad():
                seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 else seg_imgs
                seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels
                logits = model(seg_imgs)

                prepareForComputeSegMetric(model.base.seg_attention.cpu(), seg_masks.cpu(), th=0.5)

                return {"logits": logits, "labels": seg_labels}


    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_visualization(
        cfg,
        model,
        test_loader,
        classes_list,
        loss_fn,
        plotFlag = False
):
    num_classes = len(classes_list)
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("fundus_prediction.inference")
    logging._warn_preinit_stderr = 0
    logger.info("Enter inferencing")

    metrics_eval = {"overall_accuracy": Accuracy(output_transform=lambda x: (x["logits"], x["labels"])),
                    "precision": Precision(output_transform=lambda x: (x["logits"], x["labels"])),
                    "recall": Recall(output_transform=lambda x: (x["logits"], x["labels"])),
                    "confusion_matrix": ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: (x["logits"], x["labels"])),
                    #"seg_confusion_matrix": ConfusionMatrix(num_classes=1, output_transform=lambda x: (x["segmentations"], x["gtmasks"])), #会选取最大值作为预测标签，那么对于多标签有些无力
                    }
    evaluator = create_supervised_visualizer(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    y_pred = []
    y_label = []
    metrics = dict()

    @evaluator.on(Events.ITERATION_COMPLETED)
    def log_eval_step(engine):
        print("Iteration[{} / {}]".format(engine.state.iteration, len(test_loader)))


    @evaluator.on(Events.ITERATION_COMPLETED, y_pred, y_label)
    def combineTensor(engine, y_pred, y_label):
        scores = engine.state.output["logits"].cpu().numpy().tolist()
        labels = engine.state.output["labels"].cpu().numpy().tolist()
        #acc = (engine.state.output["logits"].max(1)[1] == engine.state.output["labels"]).float().mean()
        #print(acc.item())
        y_pred = y_pred.extend(scores)   #注意，此处要用extend，否则+会产生新列表
        y_label = y_label.extend(labels)


    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        precision = engine.state.metrics['precision'].numpy().tolist()
        precision_dict = {}
        avg_precision = 0
        for index, ap in enumerate(precision):
            avg_precision = avg_precision + ap
            precision_dict[index] = float("{:.3f}".format(ap))
        avg_precision = avg_precision / len(precision)
        precision_dict["avg_precision"] = float("{:.3f}".format(avg_precision))

        recall = engine.state.metrics['recall'].numpy().tolist()
        recall_dict = {}
        avg_recall = 0
        for index, ar in enumerate(recall):
            avg_recall = avg_recall + ar
            recall_dict[index] = float("{:.3f}".format(ar))
        avg_recall = avg_recall / len(recall)
        recall_dict["avg_recall"] = float("{:.3f}".format(avg_recall))

        confusion_matrix = engine.state.metrics['confusion_matrix'].numpy()

        kappa = compute_kappa(confusion_matrix)

        overall_accuracy = engine.state.metrics['overall_accuracy']
        logger.info("Test Results")
        logger.info("Precision: {}".format(precision_dict))
        logger.info("Recall: {}".format(recall_dict))
        logger.info("Overall_Accuracy: {:.3f}".format(overall_accuracy))
        logger.info("ConfusionMatrix: x-groundTruth  y-predict \n {}".format(confusion_matrix))
        logger.info("Kappa: {}".format(kappa))


        metrics["precision"] = precision_dict
        metrics["recall"] = recall_dict
        metrics["overall_accuracy"] = overall_accuracy
        metrics["confusion_matrix"] = confusion_matrix

        DF = pd.DataFrame(
            columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])  #11

        seg_class = 4  # others

        global segmentationMetric
        for layer_key in segmentationMetric.keys():
            for th_key in segmentationMetric[layer_key].keys():
                for label_key in segmentationMetric[layer_key][th_key].keys():
                    HIT = segmentationMetric[layer_key][th_key][label_key]["HIT"]
                    MISS = segmentationMetric[layer_key][th_key][label_key]["MISS"]
                    WRONG = segmentationMetric[layer_key][th_key][label_key]["WRONG"]

                    Object_Precision = HIT / (HIT + WRONG + 1E-12)
                    Object_Binary_Pre = "{:.3f}".format(Object_Precision[-1].item())
                    Object_Others_Pre = "{:.3f}".format(Object_Precision[-2].item())
                    Object_Precision = Object_Precision[0:seg_class]
                    Object_Pre = ["{:.3f}".format(Object_Precision[i].item()) for i in range(seg_class)]
                    Object_Pre_mean = "{:.3f}".format(torch.mean(Object_Precision).item())

                    Object_Recall = HIT / (HIT + MISS + 1E-12)
                    Object_Binary_Rec = "{:.3f}".format(Object_Recall[-1].item())
                    Object_Others_Rec = "{:.3f}".format(Object_Recall[-2].item())
                    Object_Recall = Object_Recall[0:seg_class]
                    Object_Rec = ["{:.3f}".format(Object_Recall[i].item()) for i in range(seg_class)]
                    Object_Rec_mean = "{:.3f}".format(torch.mean(Object_Recall).item())


                    TP = segmentationMetric[layer_key][th_key][label_key]["TP"]
                    FP = segmentationMetric[layer_key][th_key][label_key]["FP"]
                    TN = segmentationMetric[layer_key][th_key][label_key]["TN"]
                    FN = segmentationMetric[layer_key][th_key][label_key]["FN"]


                    # CJY at 2020.3.1  add seg IOU 等metrics
                    Accuracy = (TP + TN) / (TP + FP + TN + FN + 1E-12)
                    Binary_Acc = "{:.3f}".format(Accuracy[-1].item())
                    Others_Acc = "{:.3f}".format(Accuracy[-2].item())
                    Accuracy = Accuracy[0:seg_class]
                    Acc = ["{:.3f}".format(Accuracy[i].item()) for i in range(seg_class)]
                    Acc_mean = "{:.3f}".format(torch.mean(Accuracy).item())

                    Precision = TP / (TP + FP + 1E-12)
                    Binary_Pre = "{:.3f}".format(Precision[-1].item())
                    Others_Pre = "{:.3f}".format(Precision[-2].item())
                    Precision = Precision[0:seg_class]
                    Pre = ["{:.3f}".format(Precision[i].item()) for i in range(seg_class)]
                    Pre_mean = "{:.3f}".format(torch.mean(Precision).item())

                    Recall = TP / (TP + FN + 1E-12)
                    Binary_Rec = "{:.3f}".format(Recall[-1].item())
                    Others_Rec = "{:.3f}".format(Recall[-2].item())
                    Recall = Recall[0:seg_class]
                    Rec = ["{:.3f}".format(Recall[i].item()) for i in range(seg_class)]
                    Rec_mean = "{:.3f}".format(torch.mean(Recall).item())

                    IOU = TP / (TP + FP + FN + 1E-12)
                    Binary_IOU = "{:.3f}".format(IOU[-1].item())
                    Others_IOU = "{:.3f}".format(IOU[-2].item())
                    IOU = IOU[0:seg_class]
                    IU = ["{:.3f}".format(IOU[i].item()) for i in range(seg_class)]
                    IU_mean = "{:.3f}".format(torch.mean(IOU).item())

                    logger.info("Segmentation Metrics-layer-{}-th-{}-label-{}".format(layer_key, th_key, label_key))
                    logger.info("HIT   : {}".format(HIT.numpy()))
                    logger.info("MISS  : {}".format(MISS.numpy()))
                    logger.info("WRONG : {}".format(WRONG.numpy()))
                    logger.info("TP    : {}".format(TP.numpy()))
                    logger.info("FP    : {}".format(FP.numpy()))
                    logger.info("TN    : {}".format(TP.numpy()))
                    logger.info("FN    : {}".format(FP.numpy()))

                    logger.info(
                        "O_Pre    : {}, mean: {}, binary: {}, complementary: {}".format(Object_Pre, Object_Pre_mean, Object_Binary_Pre, Object_Others_Pre))
                    logger.info(
                        "O_Rec    : {}, mean: {}, binary: {}, complementary: {}".format(Object_Rec, Object_Rec_mean, Object_Binary_Rec, Object_Others_Rec))
                    logger.info(
                        "Accuracy : {}, mean: {}, binary: {}, complementary: {}".format(Acc, Acc_mean, Binary_Acc, Others_Acc))
                    logger.info(
                        "Precision: {}, mean: {}, binary: {}, complementary: {}".format(Pre, Pre_mean, Binary_Pre, Others_Pre))
                    logger.info(
                        "Recall   : {}, mean: {}, binary: {}, complementary: {}".format(Rec, Rec_mean, Binary_Rec, Others_Rec))
                    logger.info(
                        "IOU      : {}, mean: {}, binary: {}, complementary: {}".format(IU, IU_mean, Binary_IOU, Others_IOU))

                    #["visulization_method", "observation_module", "threshold", "metric", "dataset", "MA", "EX", "SE", "HE", "Others", "Mean", "Binary"])
                    DF_Object_Pre = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Object_Precision", label_key, Object_Pre[0], Object_Pre[1], Object_Pre[2], Object_Pre[3], Object_Pre_mean, Object_Binary_Pre, Object_Others_Pre]],
                                          columns=["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_Object_Rec = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Object_Recall", label_key, Object_Rec[0], Object_Rec[1], Object_Rec[2], Object_Rec[3], Object_Rec_mean, Object_Binary_Rec, Object_Others_Rec]],
                                          columns=["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])

                    DF_Acc = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Accuracy", label_key, Acc[0], Acc[1], Acc[2], Acc[3], Acc_mean, Binary_Acc, Others_Acc]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_Pre = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Precision", label_key, Pre[0], Pre[1], Pre[2], Pre[3], Pre_mean, Binary_Pre, Others_Pre]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_Rec = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Recall", label_key, Rec[0], Rec[1], Rec[2], Rec[3], Rec_mean, Binary_Rec, Others_Rec]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_IOU = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "IOU", label_key, IU[0], IU[1], IU[2], IU[3], IU_mean, Binary_IOU, Others_IOU]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])

                    DF = pd.concat([DF, DF_Object_Pre, DF_Object_Rec, DF_Acc, DF_Pre, DF_Rec, DF_IOU], ignore_index=True)



        sheet_name = "train"   # val test
        xls_filename = model.visualizer_name + ".xlsx"#os.path.join(r"D:\MIP\Experiment\MBagNet", model.visualizer_name + ".xlsx")
        if os.path.exists(xls_filename) == True:
            with pd.ExcelWriter(xls_filename, mode='a') as writer:
                DF.to_excel(writer, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(xls_filename, mode='w') as writer:
                DF.to_excel(writer, sheet_name=sheet_name)


        #DF.to_excel(model.visualizer_name + ".xlsx")

    evaluator.run(test_loader)

    # 1.Draw Confusion Matrix and Save it in numpy
    """
    # CJY at 2020.6.24
    classes_label_list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative", "Ungradable"]
    if len(classes_list) == 6:
        classes_list = classes_label_list

    confusion_matrix_numpy = drawConfusionMatrix(metrics["confusion_matrix"], classes=np.array(classes_list), title='Confusion matrix', drawFlag=False)  #此处就不画混淆矩阵了
    metrics["confusion_matrix_numpy"] = confusion_matrix_numpy
    #"""

    # 2.ROC
    """
    # (1).convert List to numpy
    y_label = np.array(y_label)
    y_label = convert_to_one_hot(y_label, num_classes)
    y_pred = np.array(y_pred)

    #注：此处可以提前将多类label转化为one-hot label，并以每一类的confidence和label sub-vector送入计算
    #不一定要送入score（概率化后的值），只要confidengce与score等是正相关即可（单调递增）

    # (2).Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    pos_label = 0   #for two classes
    if num_classes == 2:
        fpr[pos_label], tpr[pos_label], _ = roc_curve(y_label[:, pos_label], y_pred[:, pos_label])   #当y_label并非0,1组合的向量时，即多分类标签，可以通过指定pos_label=
        roc_auc[pos_label] = auc(fpr[pos_label], tpr[pos_label])
    elif num_classes > 2:
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i])
            roc_auc[i] = float("{:.3f}".format(auc(fpr[i], tpr[i])))

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
        roc_auc["micro"] = float("{:.3f}".format(auc(fpr["micro"], tpr["micro"])))

    logger.info("ROC_AUC: {}".format(roc_auc))
    metrics["roc_auc"] = roc_auc

    # (3).Draw ROC and Save it in numpy   # 好像绘制，在服务器上会出错，先取消吧
    if num_classes == 2:
        roc_numpy = plotROC_OneClass(fpr[pos_label], tpr[pos_label], roc_auc[pos_label], plot_flag=plotFlag)
    elif num_classes > 2:
        roc_numpy = plotROC_MultiClass(fpr, tpr, roc_auc, num_classes, plot_flag=plotFlag)
    metrics["roc_figure"] = roc_numpy
    #"""

    return metrics


def compute_kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)