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
        grade_labels = grade_labels.to(device) if torch.cuda.device_count() >= 1 and grade_labels is not None else grade_labels
        seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 and seg_imgs is not None else seg_imgs
        seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 and seg_masks is not None else seg_masks
        seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 and seg_labels is not None else seg_labels

        heatmapType = "visualization"
        savePath = os.path.join(r"D:\Visualization\results", model.visualizer_name)
        if os.path.exists(savePath) != True:
            os.makedirs(savePath)
        else:
            raise Exception("Folder Exists")

        # 记录grade和seg的样本数量
        grade_num = grade_imgs.shape[0] if grade_imgs is not None else 0
        seg_num = seg_masks.shape[0] if seg_imgs is not None else 0
        # 将grade和seg样本concat起来
        if grade_num > 0 and seg_num > 0:       # joint
            dataType = "joint"
            imgs = torch.cat([grade_imgs, seg_imgs], dim=0)
            labels = torch.cat([grade_labels, seg_labels], dim=0)
            masks = seg_masks
            img_paths = gimg_path + simg_path
        elif grade_num > 0 and seg_num == 0:    # grade
            dataType = "grade"
            # grade_labels  #242 boxer, 243 bull mastiff p, 281 tabby cat p,282 tiger cat, 250 Siberian husky, 333 hamster
            imgs = grade_imgs
            labels = torch.zeros_like(grade_labels) + 1  # 333#243
            #labels = grade_labels
            masks = None
            img_paths = gimg_path
        elif grade_num == 0 and seg_num > 0:    # seg
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
            # 由于需要用到梯度进行可视化计算，所以就不加入with torch.no_grad()了
            logits = model(imgs)

            if model.classifier_output_type == "multi-label":
                p_labels = (logits.relu() * model.lesion_area_std_dev).int()
            else:
                p_labels = torch.argmax(logits, dim=1)  # predict_label

            # 显示图片的数字还是原始名字
            if hasattr(engine.state, "imgsName") != True:
                engine.state.imgsName =[]
            #"""
            # 数字
            if engine.state.imgsName == []:
                engine.state.imgsName = ["{}".format(i) for i in range(imgs.shape[0])]
            else:
                engine.state.imgsName = [str(int(i)+imgs.shape[0]) for i in engine.state.imgsName]
            if int(engine.state.imgsName[0]) >= 20:
                exit(0)
            #"""
            # 名字
            #engine.state.imgsName = [os.path.split(img_path)[1].split(".")[0] for img_path in img_paths]

            # 观测类别
            if model.num_classes < 30:
                oblabelList = [labels*0 + i for i in range(model.num_classes)] if len(labels.shape)==1 else [labels.sum(1)*0 + i for i in range(model.num_classes)]
            elif model.num_classes == 1000:
                oblabelList = [labels*243, labels*250, labels*281, labels*333]
            #oblabelList = [labels]
            #oblabelList = [p_labels]
            #oblabelList = [labels, p_labels]

            # 可视化
            showFlag = 1
            for oblabels in oblabelList:
                binary_threshold = 0#0.25#0.5
                if model.visualizer.reservePos != True:
                    binary_threshold = binary_threshold*0.5 + 0.5

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
                    model.visualizer.DrawVisualization(vimgs, vlabels, vplabels, vmasks, binary_threshold, savePath, engine.state.imgsName)

                if dataType == "seg":
                    if hasattr(engine.state, "MPG")!=True:
                        engine.state.MPG = MultiPointingGame(visual_class_list=range(4), seg_class_list=range(4))

                    binary_gtmasks = torch.max(vmasks, dim=1, keepdim=True)[0]
                    gtmasks = torch.cat([vmasks, 1 - binary_gtmasks, binary_gtmasks], dim=1)
                    for i, v in enumerate(gcam_list):
                        rv = torch.nn.functional.interpolate(v, input_size, mode='bilinear')
                        segmentations = rv  # .gt(binary_threshold)
                        if segmentations.shape[1] == 1:
                            engine.state.MPG.update(segmentations.cpu(), gtmasks.cpu(), oblabels.cpu(), model.visualizer_name, model.visualizer.target_layer[i], binary_threshold)

            labels = labels if len(labels.shape) == 1 else torch.max(labels, dim=1)[1]

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