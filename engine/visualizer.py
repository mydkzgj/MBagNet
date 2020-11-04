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
from metric.multi_pointing_game import *


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

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

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

def convertMultiMask(seg_masks, num_classes):
    batches = seg_masks.shape[0]
    channels = seg_masks.shape[1]
    if channels > 1:
        raise Exception
    classwise_seg_masks_list = []
    for c in range(num_classes):
        classwise_seg_masks_list.append(seg_masks.eq(c+1))

    classwise_seg_masks = torch.cat(classwise_seg_masks_list, dim=1)
    multi_labels = classwise_seg_masks.sum(-1).sum(-1).gt(0).float()

    return classwise_seg_masks, multi_labels



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
        if grade_labels is not None and isinstance(grade_labels[0], dict): #CJY at 2020.10.18
            annotations = grade_labels
            grade_labels = [obj["multi-labels"] for obj in annotations]
            grade_labels = torch.tensor(grade_labels, dtype=torch.int64)
        elif seg_masks is not None and seg_labels is None:
            annotations = None
            seg_masks, seg_labels = convertMultiMask(seg_masks*255, model.num_classes)
        else:
            annotations = None

        grade_imgs = grade_imgs.to(device) if torch.cuda.device_count() >= 1 and grade_imgs is not None else grade_imgs
        grade_labels = grade_labels.to(device) if torch.cuda.device_count() >= 1 and grade_labels is not None else grade_labels
        seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 and seg_imgs is not None else seg_imgs
        seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 and seg_masks is not None else seg_masks
        seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 and seg_labels is not None else seg_labels

        heatmapType = "visualization"
        show_imagename_type = "number"
        run_image_maxnum = -1 if model.run_sub_dataset_name == "test" else 100
        savePath = os.path.join(r"D:\Visualization\results", model.visualizer_name)
        showFlag = 1
        max_show_num = 100
        computeMetirc = 3 if model.run_sub_dataset_name == "test" else 0
        model.visualizer.reservePos = False if model.run_sub_dataset_name == "test" else True# CJY at 2020.10.20 全局控制开关

        if engine.state.iteration == 1:
            if os.path.exists(savePath) != True:
                os.makedirs(savePath)

        # 记录grade和seg的样本数量
        grade_num = grade_imgs.shape[0] if grade_imgs is not None else 0
        seg_num = seg_masks.shape[0] if seg_imgs is not None else 0
        # 将grade和seg样本concat起来 #不考虑joint的形式
        if grade_num > 0 and seg_num == 0:    # grade
            dataType = "grade"
            imgs = grade_imgs
            labels = grade_labels
            masks = None
            img_paths = gimg_path
        elif grade_num >= 0 and seg_num > 0:    # seg
            dataType = "seg"
            imgs = seg_imgs
            labels = seg_labels
            masks = seg_masks
            img_paths = simg_path

        model.transmitClassifierWeight()  # 该函数是将baseline中的finalClassifier的weight传回给base，使得其可以直接计算logits-map，
        model.transimitBatchDistribution(1)  # 所有样本均要生成可视化seg

        if heatmapType == "segmentation":
            with torch.no_grad():
                logits = model(imgs)
                scores = torch.softmax(logits, dim=1)
                p_labels = torch.argmax(logits, dim=1)  # predict_label
                return {"logits": logits, "labels": labels}

        elif heatmapType == "visualization":
            if hasattr(model.visualizer, "multiply_input"):
                if model.visualizer.multiply_input > 1:
                    imgs = torch.cat([imgs] * model.visualizer.multiply_input, dim=0)
            # 由于需要用到梯度进行可视化计算，所以就不加入with torch.no_grad()了
            logits = model(imgs)
            if hasattr(model.visualizer, "multiply_input"):
                if model.visualizer.multiply_input > 1:
                    imgs = imgs[0:imgs.shape[0]//model.visualizer.multiply_input]

            if model.classifier_output_type == "multi-label":
                p_labels = torch.sort(logits, dim=1, descending=True)
            else:
                p_labels = torch.argmax(logits, dim=1)    # predict_label

            # 显示图片的数字还是原始名字
            start_index = (engine.state.iteration - 1) * imgs.shape[0] + 1
            engine.state.imgs_index = [start_index + i for i in range(imgs.shape[0])]
            if show_imagename_type == "name":  #名字
                engine.state.imgs_name = [os.path.split(img_path)[1].split(".")[0] for img_path in img_paths]
            elif show_imagename_type == "number":  # 数字
                engine.state.imgs_name = ["{}".format(i) for i in engine.state.imgs_index]

            if engine.state.imgs_index[0] > run_image_maxnum and run_image_maxnum != -1:
                exit(0)

            # 观测类别
            num_th = 3
            if model.num_classes == 4 and model.classifier_output_type == "multi-label":
                # DDR-LESION
                s_labels = labels
                s_p_labels = (logits.relu() * model.lesion_area_std_dev).int()
                oblabelList = [labels.sum(1)*0 + i for i in range(model.num_classes)]
            elif model.num_classes == 6 and model.classifier_output_type == "single-label":
                # DDR
                s_labels = labels
                s_p_labels = torch.argmax(logits, dim=1)
                oblabelList = [labels * 0 + i for i in range(model.num_classes)]
            elif model.num_classes == 20 and model.classifier_output_type == "multi-label":
                # PASCAL-VOC
                num_labels = labels.gt(0).sum().item()
                s_labels = torch.sort(labels, dim=1, descending=True)[1][:, 0:num_labels]
                s_p_labels = torch.sort(logits, dim=1, descending=True)[1][0:s_labels.shape[0], 0:num_labels+num_th]
                o_labels = torch.cat([s_labels, s_p_labels], dim=1)
                oblabelList = [o_labels[:, j] for j in range(o_labels.shape[1])]
            elif model.num_classes == 80 and model.classifier_output_type == "multi-label":
                # PASCAL-VOC
                num_labels = labels.gt(0).sum().item()
                s_labels = torch.sort(labels, dim=1, descending=True)[1][:, 0:num_labels]
                s_p_labels = torch.sort(logits, dim=1, descending=True)[1][0:s_labels.shape[0], 0:num_labels + num_th]
                o_labels = torch.cat([s_labels, s_p_labels], dim=1)
                oblabelList = [o_labels[:, j] for j in range(o_labels.shape[1])]
            elif model.num_classes == 1000:
                # ImageNet
                # grade_labels  #242 boxer, 243 bull mastiff p, 281 tabby cat p,282 tiger cat, 250 Siberian husky, 333 hamster
                s_labels = labels * 0 + 1
                s_p_labels = torch.argmax(logits, dim=1)
                oblabelList = [s_labels*243, s_labels*250, s_labels*281, s_labels*333]

            if computeMetirc == 1:    #观察全部类别
                s_labels = labels
                s_p_labels = torch.argmax(logits, dim=1)
                oblabelList = [labels.sum(1) * 0 + i for i in range(model.num_classes)]
            elif computeMetirc == 2:  #观察gt-label的类别
                num_labels = labels.gt(0).sum().item()
                s_labels = torch.sort(labels, dim=1, descending=True)[1][:, 0:num_labels]
                oblabelList = [s_labels[:, j] for j in range(s_labels.shape[1])]
            elif computeMetirc == 3:  #观察预测正确的gt-label的类别
                pright_labels = labels.float() * logits[0:labels.shape[0]].gt(0).float()
                num_labels = pright_labels.gt(0).sum().item()
                s_pr_labels = torch.sort(pright_labels, dim=1, descending=True)[1][:, 0:num_labels]
                oblabelList = [s_pr_labels[:, j] for j in range(s_pr_labels.shape[1])]

            #oblabelList = [labels]
            #oblabelList = [p_labels]
            #oblabelList = [labels, p_labels]

            # 可视化
            for oblabels in oblabelList:
                binary_threshold = 0.5  #0.25#0.5
                if model.visualizer.reservePos != True:
                    binary_threshold = binary_threshold * 0.5 + 0.5

                input_size = (imgs.shape[2], imgs.shape[3])
                visual_num = imgs.shape[0]
                gcam_list, gcam_max_list, overall_gcam = model.visualizer.GenerateVisualiztions(logits, oblabels, input_size, visual_num=visual_num)

                # 用于visualizaztion的数据
                vimgs = imgs[imgs.shape[0] - visual_num:imgs.shape[0]]
                vlabels = s_labels[imgs.shape[0] - visual_num:imgs.shape[0]]
                vplabels = s_p_labels[imgs.shape[0] - visual_num:imgs.shape[0]]
                vmasks = masks[masks.shape[0] - visual_num:masks.shape[0]] if masks is not None else None
                vannotations = annotations[imgs.shape[0] - visual_num:imgs.shape[0]] if annotations is not None else None

                if showFlag == 1:
                    if engine.state.imgs_index[0] < max_show_num and  max_show_num != -1:
                        # 绘制可视化结果
                        if vmasks is not None:
                            model.visualizer.DrawVisualization(vimgs, vlabels, vplabels, vmasks, binary_threshold, savePath, engine.state.imgs_name)
                        elif vannotations is not None:
                            model.visualizer.DrawVisualization(vimgs, vlabels, vplabels, vannotations, binary_threshold, savePath, engine.state.imgs_name)

                if computeMetirc != 0:
                    if dataType == "seg" and (model.num_classes == 4 or model.num_classes == 20):
                        if hasattr(engine.state, "MPG") != True:
                            engine.state.MPG = MultiPointingGameForSegmentation(visual_class_list=range(model.num_classes), seg_class_list=range(model.num_classes))

                        for i, v in enumerate(gcam_list):
                            segmentations = torch.nn.functional.interpolate(v, input_size, mode='bilinear')
                            engine.state.MPG.update(segmentations.cpu(), vmasks.cpu(), oblabels.cpu(),
                                                    model.visualizer_name, model.visualizer.target_layer[i], binary_threshold)
                    elif model.num_classes == 20 and isinstance(annotations[0], dict):  # CJY at 2020.10.18
                        if hasattr(engine.state, "MPG") != True:
                            engine.state.MPG = MultiPointingGameForDetection(visual_class_list=range(20), seg_class_list=range(20))


                        for i, v in enumerate(gcam_list):
                            segmentations = torch.nn.functional.interpolate(v, input_size, mode='bilinear')
                            engine.state.MPG.update(segmentations.cpu(), vannotations, oblabels.cpu(),
                                                    model.visualizer_name, model.visualizer.target_layer[i], binary_threshold)

            labels = labels if len(labels.shape) == 1 else torch.max(labels, dim=1)[1]

            if hasattr(model.visualizer, "multiply_input"):
                if model.visualizer.multiply_input > 1:
                    logits = logits[0: logits.shape[0]//model.visualizer.multiply_input]

            return {"logits": logits.detach(), "labels": labels, }

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

        overall_accuracy = engine.state.metrics['overall_accuracy']

        confusion_matrix = engine.state.metrics['confusion_matrix'].numpy()

        kappa = compute_kappa(confusion_matrix)

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

        # CJY at 2020.8.16 Multi-Pointing Game Save XLS
        if hasattr(engine.state, "MPG")==True:
            engine.state.MPG.saveXLS(savePath=".")

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
    return (po - pe) / max((1 - pe), 1E-12)