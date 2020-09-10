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
from ignite.metrics import MeanSquaredError

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

            grade_imgs, grade_labels, seg_imgs, seg_masks, seg_labels, gimg_path, simg_path = batch

            # 记录grade和seg的样本数量
            grade_num = grade_imgs.shape[0] if grade_imgs is not None else 0
            seg_num = seg_masks.shape[0] if seg_imgs is not None else 0
            # 将grade和seg样本concat起来
            if grade_num > 0 and seg_num >= 0:
                imgs = grade_imgs
                labels = grade_labels
            elif grade_num == 0 and seg_num > 0:
                imgs = seg_imgs
                labels = seg_labels

            imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
            labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
            # 创建multi-labels以及regression_labels
            if len(labels.shape) == 1:  # 如果本身是标量标签
                one_hot_labels = torch.nn.functional.one_hot(labels, model.num_classes).float()
                one_hot_labels = one_hot_labels.to(
                    device) if torch.cuda.device_count() >= 1 and one_hot_labels is not None else one_hot_labels
                regression_labels = 0
            else:  # 如果本身是向量标签
                one_hot_labels = torch.gt(labels, 0).int()
                regression_labels = (labels.float() - model.lesion_area_mean) / model.lesion_area_std_dev  # label 标准化

            model.transimitBatchDistribution(0)  #不生成seg
            logits = model(imgs)

            if model.classifier_output_type == "single-label":
                scores = torch.softmax(logits, dim=1)
                regression_logits = 0
            else:
                # CJY at 2020.9.5
                scores = torch.sigmoid(logits).round()
                regression_logits = model.zoom_ratio * torch.relu(logits)#model.regression_linear(logits.unsqueeze(1)).squeeze(1) #model.regression_linear(logits)

            return {"logits": logits, "scores":scores, "labels": labels, "multi-labels":one_hot_labels,
                    "regression-logits": regression_logits, "regression-labels": regression_labels}



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

    if model.classifier_output_type == "multi-label":
        #此处用了转置的话，如果batch_size为1，会出现报错（因为num_class>1 if multi-label）所以串联为为2倍。  Failed
        #此处average都选择True， 否则将会将所有sample串联记录下来。 但是在trainer中由于用了RunningAverage，所以不用average=False
        #"""
        metrics_eval = {"overall_accuracy": Accuracy(output_transform=lambda x: (x["scores"], x["multi-labels"]), is_multilabel=True),
                        #"precision": Precision(average=False, output_transform=lambda x: (torch.cat([x["logits"], x["logits"]], dim=0).sigmoid().round().transpose(1,0), torch.cat([x["labels"], x["labels"]], dim=0).transpose(1,0)), is_multilabel=True),
                        #"recall": Recall(average=False, output_transform=lambda x: (torch.cat([x["logits"], x["logits"]], dim=0).sigmoid().round().transpose(1,0), torch.cat([x["labels"], x["labels"]], dim=0).transpose(1,0)), is_multilabel=True),
                        "precision": Precision(average=True, output_transform=lambda x: (x["scores"], x["multi-labels"]), is_multilabel=True),
                        "recall": Recall(average=True, output_transform=lambda x: (x["scores"], x["multi-labels"]), is_multilabel=True),
                        "mse": MeanSquaredError(output_transform=lambda x: (x["regression-logits"], x["regression-labels"])),
                        "confusion_matrix": ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: (x["logits"], torch.max(x["labels"], dim=1)[1])),
                        }
        #"""

        """
        # From Github Ignite. Answer for my first issue!!!! 
        # Setup label-wise metrics
        # -> let's assume that evaluator returns output as (y_pred, y)
        # -> y_pred.shape: (B, C) and y.shape: (B, C)
        def get_single_label_output_fn(c):
            def wrapper(output):
                y_pred = output["logits"]
                y = output["labels"]
                return y_pred[:, c].sigmoid().round(), y[:, c]

            return wrapper

        metrics_eval = {}
        for i in range(4):
            for name, cls in zip(["Accuracy", "Precision", "Recall"], [Accuracy, Precision, Recall]):
                metrics_eval["{}/{}".format(name, i)] = cls(output_transform=get_single_label_output_fn(i))
                #metrics_eval["{}/{}".format(name, i)] = cls(output_transform=lambda x: (x["logits"][:,i].sigmoid().round(), x["labels"][:,i]))
        #"""

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
        precision_dict = {}
        if isinstance(engine.state.metrics['precision'], torch.Tensor):
            precision = engine.state.metrics['precision'].numpy().tolist()
            avg_precision = 0
            for index, ap in enumerate(precision):
                avg_precision = avg_precision + ap
                precision_dict[index] = float("{:.3f}".format(ap))
            avg_precision = avg_precision / len(precision)
            precision_dict["avg_precision"] = float("{:.3f}".format(avg_precision))
        else:
            precision_dict["avg_precision"] = float("{:.3f}".format(engine.state.metrics['precision']))

        recall_dict = {}
        if isinstance(engine.state.metrics['recall'], torch.Tensor):
            recall = engine.state.metrics['recall'].numpy().tolist()
            avg_recall = 0
            for index, ar in enumerate(recall):
                avg_recall = avg_recall + ar
                recall_dict[index] = float("{:.3f}".format(ar))
            avg_recall = avg_recall / len(recall)
            recall_dict["avg_recall"] = float("{:.3f}".format(avg_recall))
        else:
            recall_dict["avg_recall"] = float("{:.3f}".format(engine.state.metrics['recall']))

        confusion_matrix = engine.state.metrics['confusion_matrix'].numpy()

        kappa = compute_kappa(confusion_matrix)

        overall_accuracy = engine.state.metrics['overall_accuracy']
        logger.info("Test Results")
        logger.info("Precision: {}".format(precision_dict))
        logger.info("Recall: {}".format(recall_dict))
        logger.info("Overall_Accuracy: {:.3f}".format(overall_accuracy))
        if engine.state.metrics.get("mse") != None:
            mse = engine.state.metrics["mse"]
            logger.info("MSE: {:.3f}".format(mse))
        logger.info("ConfusionMatrix: x-groundTruth  y-predict \n {}".format(confusion_matrix))
        logger.info("Kappa: {}".format(kappa))

        metrics["precision"] = precision_dict
        metrics["recall"] = recall_dict
        metrics["overall_accuracy"] = overall_accuracy
        metrics["confusion_matrix"] = confusion_matrix

    evaluator.run(test_loader)

    if plotFlag == True:  # 绘制   CJY at 2020.8.3
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