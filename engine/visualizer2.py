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


segmentationMetric = {}  # 用于保存TP,FP,TN,FN

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def prepareForComputeSegMetric(seg_map, seg_mask, label_name, layer_name, th=0.5):   # 适用于多标签
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

    global segmentationMetric
    if segmentationMetric.get(layer_name) == None:
        segmentationMetric[layer_name] = {}

    if segmentationMetric[layer_name].get(th) == None:
        segmentationMetric[layer_name][th] = {}

    if segmentationMetric[layer_name][th].get(label_name) == None:
        segmentationMetric[layer_name][th][label_name] = {"TP":0, "FP":0, "TN":0, "FN":0}

    segmentationMetric[layer_name][th][label_name]["TP"] = segmentationMetric[layer_name][th][label_name]["TP"] + tp
    segmentationMetric[layer_name][th][label_name]["FP"] = segmentationMetric[layer_name][th][label_name]["FP"] + fp
    segmentationMetric[layer_name][th][label_name]["TN"] = segmentationMetric[layer_name][th][label_name]["TN"] + tn
    segmentationMetric[layer_name][th][label_name]["FN"] = segmentationMetric[layer_name][th][label_name]["FN"] + fn

    return tp, fp, tn, fn

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
        grade_imgs, grade_labels, seg_imgs, seg_masks, seg_labels = batch
        grade_imgs = grade_imgs.to(device) if torch.cuda.device_count() >= 1 else grade_imgs
        grade_labels = grade_labels.to(device) if torch.cuda.device_count() >= 1 else grade_labels
        seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 else seg_imgs
        seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 else seg_masks
        seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels

        model.transmitClassifierWeight()  # 该函数是将baseline中的finalClassifier的weight传回给base，使得其可以直接计算logits-map，
        model.transimitBatchDistribution(1)  # 所有样本均要生成可视化seg

        dataType = "grade"
        heatmapType = "visualization"  # "GradCAM"#"segmenters"#"GradCAM"#"computeSegMetric"  # "grade", "segmenters", "computeSegMetric", "GradCAM"
        savePath = r"D:\MIP\Experiment\1"

        if dataType == "grade":
            imgs = grade_imgs
            labels = grade_labels
            masks = torch.zeros_like(imgs)
        elif dataType == "seg":
            imgs = seg_imgs
            labels = seg_labels
            masks = seg_masks
        elif dataType == "joint":
            imgs = torch.cat([grade_imgs, seg_imgs], dim=0)
            labels = torch.cat([grade_labels, seg_labels], dim=0)
            masks = seg_masks

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

            #if engine.state.iteration > 10:
            #   return {"logits": logits.detach(), "labels": seg_labels, }

            #lesionsStatistics(seg_masks, seg_labels.item())

            binary_threshold = 0.5
            showFlag = 1
            input_size = (imgs.shape[2], imgs.shape[3])
            visualBD = [grade_imgs.shape[0], grade_imgs.shape[0]]#seg_imgs.shape[0]]
            gcam_list, gcam_max_list, overall_gcam = model.visualizer.GenerateVisualiztions(logits, labels, input_size, visual_num=visualBD[1])

            # 用于visualizaztion的数据
            vimgs = imgs[imgs.shape[0] - visualBD[1]:imgs.shape[0]]
            vlabels = labels[imgs.shape[0] - visualBD[1]:imgs.shape[0]]
            vplabels = p_labels[imgs.shape[0] - visualBD[1]:imgs.shape[0]]
            vmasks = masks[masks.shape[0] - visualBD[1]:masks.shape[0]]

            if showFlag ==1:
                model.visualizer.DrawVisualization(vimgs, vlabels, vplabels, vmasks, binary_threshold, savePath)

            if dataType == "seg":
                binary_gtmasks = torch.max(vmasks, dim=1, keepdim=True)[0]
                gtmasks = torch.cat([vmasks, 1 - binary_gtmasks, binary_gtmasks], dim=1)
                for i, v in enumerate(gcam_list):
                    rv = torch.nn.functional.interpolate(v, input_size, mode='bilinear')
                    segmentations = rv  # .gt(binary_threshold)
                    prepareForComputeSegMetric(segmentations.cpu(), gtmasks.cpu(), "all",
                                               layer_name=model.visualizer.target_layer[i], th=binary_threshold)
                    prepareForComputeSegMetric(segmentations.cpu(), gtmasks.cpu(),
                                               labels[imgs.shape[0] - visualBD[1]:imgs.shape[0]].item(),
                                               layer_name=model.visualizer.target_layer[i], th=binary_threshold)
            return {"logits": logits.detach(), "labels": labels,}

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

        global segmentationMetric
        for layer_key in segmentationMetric.keys():
            for th_key in segmentationMetric[layer_key].keys():
                for label_key in segmentationMetric[layer_key][th_key].keys():
                    TP = segmentationMetric[layer_key][th_key][label_key]["TP"]
                    FP = segmentationMetric[layer_key][th_key][label_key]["FP"]
                    TN = segmentationMetric[layer_key][th_key][label_key]["TN"]
                    FN = segmentationMetric[layer_key][th_key][label_key]["FN"]
                    seg_class = 4 #others
                    # CJY at 2020.3.1  add seg IOU 等metrics
                    Accuracy = (TP + TN) / (TP + FP + TN + FN + 1E-12)
                    Binary_Acc = "{:.3f}".format(Accuracy[:, -1].mean().item())
                    Others_Acc = "{:.3f}".format(Accuracy[:, -2].mean().item())
                    Accuracy = Accuracy[:, 0:seg_class]
                    Acc = ["{:.3f}".format(Accuracy[0][i].item()) for i in range(seg_class)]
                    Acc_mean = torch.mean(Accuracy).item()

                    Precision = TP / (TP + FP + 1E-12)
                    Binary_Pre = "{:.3f}".format(Precision[:, -1].mean().item())
                    Others_Pre = "{:.3f}".format(Precision[:, -2].mean().item())
                    Precision = Precision[:, 0:seg_class]
                    Pre = ["{:.3f}".format(Precision[0][i].item()) for i in range(seg_class)]
                    Pre_mean = torch.mean(Precision).item()

                    Recall = TP / (TP + FN + 1E-12)
                    Binary_Rec = "{:.3f}".format(Recall[:, -1].mean().item())
                    Others_Rec = "{:.3f}".format(Recall[:, -2].mean().item())
                    Recall = Recall[:, 0:seg_class]
                    Rec = ["{:.3f}".format(Recall[0][i].item()) for i in range(seg_class)]
                    Rec_mean = torch.mean(Recall).item()

                    IOU = TP / (TP + FP + FN + 1E-12)
                    Binary_IOU = "{:.3f}".format(IOU[:, -1].mean().item())
                    Others_IOU = "{:.3f}".format(IOU[:, -2].mean().item())
                    IOU = IOU[:, 0:seg_class]
                    IU = ["{:.3f}".format(IOU[0][i].item()) for i in range(seg_class)]
                    IU_mean = torch.mean(IOU).item()

                    logger.info("Segmentation Metrics-layer-{}-th-{}-label-{}".format(layer_key, th_key, label_key))
                    #logger.info("TP       : {}".format(TP.cpu().numpy().tolist()))
                    #logger.info("FP       : {}".format(FP.cpu().numpy().tolist()))
                    #logger.info("TN       : {}".format(TN.cpu().numpy().tolist()))
                    #logger.info("FN       : {}".format(FN.cpu().numpy().tolist()))
                    logger.info(
                        "Accuracy : {}, mean: {}, binary: {}, complementary: {}".format(Acc, Acc_mean, Binary_Acc, Others_Acc))
                    logger.info(
                        "Precision: {}, mean: {}, binary: {}, complementary: {}".format(Pre, Pre_mean, Binary_Pre, Others_Pre))
                    logger.info(
                        "Recall   : {}, mean: {}, binary: {}, complementary: {}".format(Rec, Rec_mean, Binary_Rec, Others_Rec))
                    logger.info(
                        "IOU      : {}, mean: {}, binary: {}, complementary: {}".format(IU, IU_mean, Binary_IOU, Others_IOU))

                    #["visulization_method", "observation_module", "threshold", "metric", "dataset", "MA", "EX", "SE", "HE", "Others", "Mean", "Binary"])
                    DF_Acc = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Accuracy", label_key, Acc[0], Acc[1], Acc[2], Acc[3], Acc_mean, Binary_Acc, Others_Acc]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_Pre = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Precision", label_key, Pre[0], Pre[1], Pre[2], Pre[3], Pre_mean, Binary_Pre, Others_Pre]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_Rec = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "Recall", label_key, Rec[0], Rec[1], Rec[2], Rec[3], Rec_mean, Binary_Rec, Others_Rec]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])
                    DF_IOU = pd.DataFrame([[model.visualizer_name, layer_key, th_key, "IOU", label_key, IU[0], IU[1], IU[2], IU[3], IU_mean, Binary_IOU, Others_IOU]],
                                          columns = ["Visulization Method", "Observation Module", "Threshold", "Metric", "Dataset", "MA", "EX", "SE", "HE", "Mean", "Binary", "Complementary"])

                    DF = pd.concat([DF, DF_Acc, DF_Pre, DF_Rec, DF_IOU], ignore_index=True)



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
    #"""
    # CJY at 2020.6.24
    classes_label_list = ["No DR", "Mild", "Moderate", "Severe", "Proliferative", "Ungradable"]
    if len(classes_list) == 6:
        classes_list = classes_label_list

    confusion_matrix_numpy = drawConfusionMatrix(metrics["confusion_matrix"], classes=np.array(classes_list), title='Confusion matrix', drawFlag=True)
    metrics["confusion_matrix_numpy"] = confusion_matrix_numpy
    #"""

    # 2.ROC
    #"""
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