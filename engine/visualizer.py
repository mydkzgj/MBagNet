# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

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


TP = 0
FP = 0
TN = 0
FN = 0

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def computeIOU(seg_map, seg_mask, th=0.5):
    seg_pmask = torch.gt(torch.sigmoid(seg_map), th)
    seg_mask = seg_mask.bool()

    tp = (seg_pmask & seg_mask).sum(-1).sum(-1)
    fp = (seg_pmask & (~seg_mask)).sum(-1).sum(-1)
    tn = ((~seg_pmask) & (~seg_mask)).sum(-1).sum(-1)
    fn = ((~seg_pmask) & seg_mask).sum(-1).sum(-1)
    return tp, fp, tn, fn


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
        with torch.no_grad():
            imgs, labels, seg_imgs, seg_masks, seg_labels = batch
            model.transmitClassifierWeight()  # 该函数是将baseline中的finalClassifier的weight传回给base，使得其可以直接计算logits-map，
            model.transimitBatchDistribution(1)  #所有样本均要生成可视化seg
            model.heatmapType = "GradCAM"#"segmentation"#"GradCAM"#"computeSegMetric"  # "grade", "segmentation", "computeSegMetric", "GradCAM"

            if model.heatmapType == "grade":
                imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
                labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
                logits = model(imgs)
                p_labels = torch.argmax(logits, dim=1)  # predict_label
                if model.segmentationType == "denseFC":
                    model.base.showDenseFCMask(model.base.seg_attention, imgs, labels, p_labels,)
                elif model.segmentationType == "bagFeature":
                    model.base.showRFlogitMap(model.base.rf_logits_reserve, imgs, labels, p_labels, )
                return {"logits": logits, "labels": labels}

            elif model.heatmapType == "segmentation":
                seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 else seg_imgs
                seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels
                logits = model(seg_imgs)
                p_labels = torch.argmax(logits, dim=1)  # predict_label
                if model.segmentationType == "denseFC":
                    model.base.showDenseFCMask(model.base.seg_attention, seg_imgs, seg_labels, p_labels, masklabels=seg_masks)
                elif model.segmentationType == "bagFeature":
                    model.base.showRFlogitMap(model.base.rf_logits_reserve, seg_imgs, seg_labels, p_labels, masklabels=seg_masks)
                return {"logits": logits, "labels": seg_labels}

            elif model.heatmapType == "computeSegMetric":
                seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 else seg_imgs
                seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels
                logits = model(seg_imgs)
                global TP, FP, TN, FN
                tps, fps, tns, fns = computeIOU(model.base.seg_attention.cpu(), seg_masks.cpu(), th=0.5)
                TP = TP + tps
                FP = FP + fps
                TN = TN + tns
                FN = FN + fns
                return {"logits": logits, "labels": seg_labels}

        if model.heatmapType == "GradCAM":
            model.transimitBatchDistribution(0)  # 所有样本均要生成可视化seg
            #seg_imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            #seg_labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
            seg_imgs = seg_imgs.to(device) if torch.cuda.device_count() >= 1 else seg_imgs
            seg_labels = seg_labels.to(device) if torch.cuda.device_count() >= 1 else seg_labels
            seg_masks = seg_masks.to(device) if torch.cuda.device_count() >= 1 else seg_masks

            #logits2 = model(seg_imgs)

            """
            soft_mask = seg_masks
            soft_mask = model.lesionFusion(soft_mask, seg_labels[seg_labels.shape[0] - soft_mask.shape[0]:seg_labels.shape[0]])
            max_kernel_size = 60#20#random.randint(30, 240)
            soft_mask = torch.nn.functional.max_pool2d(soft_mask, kernel_size=max_kernel_size * 2 + 1, stride=1, padding=max_kernel_size)
            rimgs = seg_imgs
            rimg_mean = rimgs.mean(-1, keepdim=True).mean(-2, keepdim=True)
            mean = torch.Tensor([[0.485, 0.456, 0.406]]).unsqueeze(-1).unsqueeze(-1).cuda()
            std = torch.Tensor([[0.229, 0.224, 0.225]]).unsqueeze(-1).unsqueeze(-1).cuda()
            rimg_fill = (torch.rand_like(rimgs)-mean)/std
            pos_masked_img = soft_mask * rimgs #+ (1 - soft_mask) * rimg_fill
            neg_masked_img = (1 - soft_mask) * rimgs# + soft_mask * rimg_mean
            seg_imgs = pos_masked_img#pos_masked_img#
            seg_masks = soft_mask
            #"""

            with torch.no_grad():
                logits = model(seg_imgs)
                scores = torch.softmax(logits, dim=-1)
                p_labels = torch.argmax(logits, dim=1)  # predict_label
            """
            target_layers = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]#["denseblock1", "denseblock2", "denseblock3", "denseblock4"]#"denseblock4" # "transition2.pool")#"denseblock3.denselayer8.relu2")#"conv0")
            if seg_labels[0] != p_labels[0]:
                fv.showGradCAM(model, seg_imgs, seg_labels, p_labels, scores, target_layers=target_layers, mask=seg_masks[0])
            #"""
            #"""["denseblock4"]#
            target_layers = ["denseblock4"]#["", "denseblock1", "denseblock2", "denseblock3", "denseblock4"]#["denseblock4"]#["denseblock1", "denseblock2", "denseblock3", "denseblock4"]#"denseblock4" # "transition2.pool")#"denseblock3.denselayer8.relu2")#"conv0")
            if 1:
                #copy.deepcopy(model)
                fv.showGradCAM(model, seg_imgs, seg_labels, p_labels, scores, target_layers=target_layers, mask=seg_masks[0])
            #"""

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
                    }
    evaluator = create_supervised_visualizer(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    y_pred = []
    y_label = []
    metrics = dict()

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

        overall_accuracy = engine.state.metrics['overall_accuracy']
        logger.info("Test Results")
        logger.info("Precision: {}".format(precision_dict))
        logger.info("Recall: {}".format(recall_dict))
        logger.info("Overall_Accuracy: {:.3f}".format(overall_accuracy))
        logger.info("ConfusionMatrix: x-groundTruth  y-predict \n {}".format(confusion_matrix))


        metrics["precision"] = precision_dict
        metrics["recall"] = recall_dict
        metrics["overall_accuracy"] = overall_accuracy
        metrics["confusion_matrix"] = confusion_matrix

        if model.heatmapType == "computeSegMetric":
            global TP, FP, TN, FN
            # CJY at 2020.3.1  add seg IOU 等metrics
            Accuracy = (TP + TN) / (TP + FP + TN + FN + 1E-12)
            Acc = [Accuracy[0][i].item() for i in range(4)]
            Acc_mean = (Acc[0] + Acc[1] + Acc[2] + Acc[3]) / 4

            Precision = TP / (TP + FP + 1E-12)
            Pre = [Precision[0][i].item() for i in range(4)]
            Pre_mean = (Pre[0] + Pre[1] + Pre[2] + Pre[3]) / 4

            Recall = TP / (TP + FN + 1E-12)
            Rec = [Recall[0][i].item() for i in range(4)]
            Rec_mean = (Rec[0] + Rec[1] + Rec[2] + Rec[3]) / 4

            IOU = TP / (TP + FP + FN + 1E-12)
            IU = [IOU[0][i].item() for i in range(4)]
            IU_mean = (IU[0] + IU[1] + IU[2] + IU[3]) / 4

            logger.info("Segmentation Metrics")
            logger.info(
                "Accuracy : {:.3f} {:.3f} {:.3f} {:.3f}, mean: {:.3f}".format(Acc[0], Acc[1], Acc[2], Acc[3], Acc_mean))
            logger.info(
                "Precision: {:.3f} {:.3f} {:.3f} {:.3f}, mean: {:.3f}".format(Pre[0], Pre[1], Pre[2], Pre[3], Pre_mean))
            logger.info(
                "Recall   : {:.3f} {:.3f} {:.3f} {:.3f}, mean: {:.3f}".format(Rec[0], Rec[1], Rec[2], Rec[3], Rec_mean))
            logger.info(
                "IOU      : {:.3f} {:.3f} {:.3f} {:.3f}, mean: {:.3f}".format(IU[0], IU[1], IU[2], IU[3], IU_mean))


    evaluator.run(test_loader)
    # Draw ConfusionMatrix
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    a = pd.DataFrame(metrics["confusion_matrix"], columns=classes_list, index=classes_list)
    ax = sns.heatmap(a, annot=True)
    ax.set_xlabel("Predict label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix")
    plt.savefig("ConfusionMatrix.png", dpi=300)
    plt.show()
    plt.close()
    """

    confusion_matrix_numpy = drawConfusionMatrix(metrics["confusion_matrix"], classes=np.array(classes_list), title='Confusion matrix')
    metrics["confusion_matrix_numpy"] = confusion_matrix_numpy


    # Plot ROC
    # convert List to numpy
    y_label = np.array(y_label)
    y_label = convert_to_one_hot(y_label, num_classes)
    y_pred = np.array(y_pred)

    #注：此处可以提前将多类label转化为one-hot label，并以每一类的confidence和label sub-vector送入计算
    #不一定要送入score（概率化后的值），只要confidengce与score等是正相关即可（单调递增）

    # Compute ROC curve and ROC area for each class
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

    if num_classes == 2:
        roc_numpy = plotROC_OneClass(fpr[pos_label], tpr[pos_label], roc_auc[pos_label], plot_flag=plotFlag)
    elif num_classes > 2:
        roc_numpy = plotROC_MultiClass(fpr, tpr, roc_auc, num_classes, plot_flag=plotFlag)

    metrics["roc_auc"] = roc_auc
    metrics["roc_figure"] = roc_numpy

    return metrics