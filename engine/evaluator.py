# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
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

def create_supervised_evaluator(model, metrics, loss_fn, device=None):
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

            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels

            model.transimitBatchDistribution(0)
            logits = model(imgs)

            return {"logits": logits, "labels": labels}



    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_inference(
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
    evaluator = create_supervised_evaluator(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    y_pred = []
    y_label = []
    metrics = dict()

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

    evaluator.run(test_loader)

    # 1.Draw Confusion Matrix and Save it in numpy
    """
    confusion_matrix_numpy = drawConfusionMatrix(metrics["confusion_matrix"], classes=np.array(classes_list), title='Confusion matrix')
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

