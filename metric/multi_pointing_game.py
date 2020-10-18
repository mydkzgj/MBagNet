# Multi Point Game

import torch
import cv2 as cv
import numpy as np
import pandas as pd
import os

class MultiPointingGameForSegmentation(object):
    def __init__(self, visual_class_list, seg_class_list):
        """
        :param visual_class_list: 纵轴
        :param seg_class_list: 横轴
        """
        self.visual_class_list = visual_class_list
        self.seg_class_list = seg_class_list
        self.num_visual_class = len(self.visual_class_list)
        self.num_seg_class = len(self.seg_class_list)

        self.elemnet_name_list = ["OBJECT_HIT", "OBJECT_MISS", "PIXEL_TP", "PIXEL_FP", "PIXEL_TN", "PIXEL_FN", "NUM_IMAGE"]
        self.metric_name_list = ["OBJECT_PRECISION", "OBJECT_RECALL", "PIXEL_ACCURACY", "PIXEL_PRECISION", "PIXEL_RECALL", "PIXEL_IOU"]

        self.state = {}

    def createInitialNumpy(self):
        initial_numpy = np.zeros((self.num_visual_class, self.num_seg_class))
        return initial_numpy

    def createInitialElement(self):
        initial_element = {"OBJECT_HIT": self.createInitialNumpy(), "OBJECT_MISS": self.createInitialNumpy(),
                           "PIXEL_TP": self.createInitialNumpy(), "PIXEL_FP": self.createInitialNumpy(),
                           "PIXEL_TN": self.createInitialNumpy(), "PIXEL_FN": self.createInitialNumpy(),
                           "NUM_IMAGE": self.createInitialNumpy()}
        return initial_element

    def createInitialMetric(self):
        initial_metric = {"OBJECT_PRECISION": self.createInitialNumpy(), "OBJECT_RECALL": self.createInitialNumpy(),
                          "PIXEL_ACCURACY": self.createInitialNumpy(), "PIXEL_PRECISION": self.createInitialNumpy(),
                          "PIXEL_RECALL": self.createInitialNumpy(), "PIXEL_IOU": self.createInitialNumpy()}
        return initial_metric


    # CJY at 2020.10.17
    def update(self, saliency_maps, seg_gtmasks, visual_labels, visualizer_name, layer_name, threshold=0.5):
        """
        :param saliency_maps: b, 1, w, h
        :param seg_gtmasks: b, c, w, h
        :param visual_labels: c
        :param visualizer_name:
        :param layer_name:
        :param threshold:
        :return:
        """
        # 1.生成二值化关注图&真值图
        if saliency_maps.shape[1] == 1:
            saliency_maps = saliency_maps.expand_as(seg_gtmasks)
        else:
            raise Exception("The shape[1] of saliency_map isn't 1.")

        binary_saliency_maps = torch.gt(saliency_maps, threshold)
        binary_seg_gtmasks = seg_gtmasks.bool()
        num_batch = saliency_maps.shape[0]

        # 2.
        for b in range(num_batch):
            key = "{}+{}+{}".format(visualizer_name, layer_name, threshold)
            if self.state.get(key) == None:
                self.state[key] = {"ELEMENT_OVERALL": self.createInitialElement(),
                                   "METRIC_OVERALL": self.createInitialMetric(),
                                   "METRIC_IMAGEWISE": self.createInitialMetric()}
            v_c_index = visual_labels[b]

            te = self.createInitialElement()
            tm = self.createInitialMetric()

            for s_c_index in range(self.num_seg_class):
                # 计算mask 转numpy
                pt_mask = binary_saliency_maps[b][s_c_index].numpy().astype(np.uint8)   #.permute(1, 2, 0)
                gt_mask = binary_seg_gtmasks[b][s_c_index].numpy().astype(np.uint8)

                hit_mask = self.fillHoles(pt_mask, gt_mask)  # 保留的是二者相交的gt那部分
                miss_mask = gt_mask - hit_mask

                # 计算mask->scalar  通过联通域检测转化为标量
                hit_retval, hit_labels, hit_stats, hit_centroids = cv.connectedComponentsWithStats(hit_mask, connectivity=8, ltype=cv.CV_32S)
                object_hit = hit_retval - 1
                miss_retval, miss_labels, miss_stats, miss_centroids = cv.connectedComponentsWithStats(miss_mask, connectivity=8, ltype=cv.CV_32S)
                object_miss = miss_retval - 1

                pixel_tp = (pt_mask * gt_mask).sum(axis=(0, 1))
                pixel_fp = (pt_mask * (1 - gt_mask)).sum(axis=(0, 1))
                pixel_tn = ((1 - pt_mask) * (1 - gt_mask)).sum(axis=(0, 1))
                pixel_fn = ((1 - pt_mask) & gt_mask).sum(axis=(0, 1))

                te["OBJECT_HIT"][v_c_index][s_c_index] = object_hit
                te["OBJECT_MISS"][v_c_index][s_c_index] = object_miss
                te["PIXEL_TP"][v_c_index][s_c_index] = pixel_tp
                te["PIXEL_FP"][v_c_index][s_c_index] = pixel_fp
                te["PIXEL_TN"][v_c_index][s_c_index] = pixel_tn
                te["PIXEL_FN"][v_c_index][s_c_index] = pixel_fn
                te["NUM_IMAGE"][v_c_index][s_c_index] = 1

            tm["OBJECT_PRECISION"] = te["OBJECT_HIT"] / np.maximum(te["OBJECT_HIT"].sum(axis=1, keepdims=True), 1E-12)
            tm["OBJECT_RECALL"] = te["OBJECT_HIT"] / np.maximum((te["OBJECT_HIT"]+te["OBJECT_MISS"]), 1E-12)
            tm["PIXEL_ACCURACY"] = (te["PIXEL_TP"]+te["PIXEL_TN"])/np.maximum((te["PIXEL_TP"]+te["PIXEL_FP"]+te["PIXEL_TN"]+te["PIXEL_FN"]), 1E-12)
            tm["PIXEL_PRECISION"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FP"]), 1E-12)
            tm["PIXEL_RECALL"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FN"]), 1E-12)
            tm["PIXEL_IOU"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FP"] + te["PIXEL_FN"]), 1E-12)

            # 记录
            for k in self.state[key]["METRIC_IMAGEWISE"].keys():
                self.state[key]["METRIC_IMAGEWISE"][k] = (self.state[key]["METRIC_IMAGEWISE"][k] * self.state[key]["ELEMENT_OVERALL"]["NUM_IMAGE"] + tm[k])\
                                                         /np.maximum((self.state[key]["ELEMENT_OVERALL"]["NUM_IMAGE"] + te["NUM_IMAGE"]), 1E-12)

            for k in self.state[key]["ELEMENT_OVERALL"].keys():
                self.state[key]["ELEMENT_OVERALL"][k] += te[k]

            self.state[key]["METRIC_OVERALL"]["OBJECT_PRECISION"] = self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] / np.maximum(self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"].sum(axis=1, keepdims=True), 1E-12)
            self.state[key]["METRIC_OVERALL"]["OBJECT_RECALL"] = self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] + self.state[key]["ELEMENT_OVERALL"]["OBJECT_MISS"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_ACCURACY"] = (self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_TN"]) \
                                                                   / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_TN"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_PRECISION"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_RECALL"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_IOU"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)


    def fillHoles(self, seedImg, Mask):
        # 输入是numpy格式
        seedImg = seedImg * Mask
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        i = 0
        while 1:
            dilated = cv.dilate(seedImg, kernel)  # 膨胀图像
            outputImg = dilated & Mask
            i = i + 1
            if i % 30 == 0:
                if (seedImg == outputImg).all():
                    break
                else:
                    seedImg = outputImg
            else:
                seedImg = outputImg
        return outputImg

    def saveXLS(self, savePath):
        # 1.生成总表，详细记录每一项
        sheetName = "SUMMARY-CLASSWISE"
        op_list = ["Visulization Method", "Observation Module", "Threshold",
                   "Metric Type", "Metric Name", "Visual Label", "Segmentation Label", "Value"]
        DF = pd.DataFrame(columns=op_list)

        for key in self.state:
            method_key, layer_key, th_key = key.split("+")
            for metric_type in self.state[key].keys():
                for metric_name in self.state[key][metric_type].keys():
                    metric_numpy = self.state[key][metric_type][metric_name]
                    for vi, vclass in enumerate(self.visual_class_list):
                        for si, sclass in enumerate(self.seg_class_list):
                            DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                        vclass, sclass, metric_numpy[vi][si]]], columns=op_list)
                            DF = pd.concat([DF, DF_NewLine], ignore_index=True)


        xls_filename = os.path.join(savePath, method_key + ".xlsx")
        """
        if os.path.exists(xls_filename) == True:
            with pd.ExcelWriter(xls_filename, mode='a') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        else:
            with pd.ExcelWriter(xls_filename, mode='w') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        """
        with pd.ExcelWriter(xls_filename, mode='w') as writer:
            DF.to_excel(writer, sheet_name=sheetName)

        #DF.to_excel(model.visualizer_name + ".xlsx")

        # 2.生成总表，详细记录每一项
        if self.visual_class_list == self.seg_class_list:
            sheetName = "SUMMARY"
            op_list = ["Visulization Method", "Observation Module", "Threshold",
                       "Metric Type", "Metric Name", "Visual Label", "Value"]
            DF = pd.DataFrame(columns=op_list)

            for key in self.state:
                method_key, layer_key, th_key = key.split("+")
                for metric_type in self.state[key].keys():
                    for metric_name in self.state[key][metric_type].keys():
                        metric_numpy = self.state[key][metric_type][metric_name]
                        mean_value = 0
                        for vi, vclass in enumerate(self.visual_class_list):
                            for si, sclass in enumerate(self.seg_class_list):
                                if vi == si:
                                    mean_value = mean_value + metric_numpy[vi][si]
                                    DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                                vclass, metric_numpy[vi][si]]], columns=op_list)
                                    DF = pd.concat([DF, DF_NewLine], ignore_index=True)

                        mean_value = mean_value / len(self.visual_class_list)
                        DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                    "mean", mean_value]], columns=op_list)
                        DF = pd.concat([DF, DF_NewLine], ignore_index=True)

            xls_filename = os.path.join(savePath, method_key + ".xlsx")
            """
            if os.path.exists(xls_filename) == True:
                with pd.ExcelWriter(xls_filename, mode='a') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            else:
                with pd.ExcelWriter(xls_filename, mode='w') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            """
            if os.path.exists(xls_filename) == True:
                with pd.ExcelWriter(xls_filename, mode='a') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            else:
                with pd.ExcelWriter(xls_filename, mode='w') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)


        #2.记录统计后的表
        sheetName = "VISUAL-LABEL"
        op_list = ["Visulization Method", "Observation Module", "Threshold", "Visual Label",] + self.metric_name_list

        DF = pd.DataFrame(columns=op_list)

        for key in self.state:
            method_key, layer_key, th_key = key.split("+")
            for vi, vclass in enumerate(self.visual_class_list):
                line = [method_key, layer_key, th_key, vclass]
                for metric_name in self.metric_name_list:
                    metric_numpy = self.state[key]["METRIC_OVERALL"][metric_name]
                    line.append(metric_numpy[vi][vi])
                DF_NewLine = pd.DataFrame([line], columns=op_list)
                DF = pd.concat([DF, DF_NewLine], ignore_index=True)

        xls_filename = os.path.join(savePath, method_key + ".xlsx")
        if os.path.exists(xls_filename) == True:
            with pd.ExcelWriter(xls_filename, mode='a') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        else:
            with pd.ExcelWriter(xls_filename, mode='w') as writer:
                DF.to_excel(writer, sheet_name=sheetName)


class MultiPointingGameForDetection(object):
    def __init__(self, visual_class_list, seg_class_list):
        """
        :param visual_class_list: 纵轴
        :param seg_class_list: 横轴
        """
        self.visual_class_list = visual_class_list
        self.seg_class_list = seg_class_list
        self.num_visual_class = len(self.visual_class_list)
        self.num_seg_class = len(self.seg_class_list)

        self.elemnet_name_list = ["OBJECT_HIT", "OBJECT_MISS", "PIXEL_TP", "PIXEL_FP", "PIXEL_TN", "PIXEL_FN", "NUM_IMAGE"]
        self.metric_name_list = ["OBJECT_PRECISION", "OBJECT_RECALL", "PIXEL_ACCURACY", "PIXEL_PRECISION", "PIXEL_RECALL", "PIXEL_IOU"]

        self.state = {}

    def createInitialNumpy(self):
        initial_numpy = np.zeros((self.num_visual_class, self.num_seg_class))
        return initial_numpy

    def createInitialElement(self):
        initial_element = {"OBJECT_HIT": self.createInitialNumpy(), "OBJECT_MISS": self.createInitialNumpy(),
                           "PIXEL_TP": self.createInitialNumpy(), "PIXEL_FP": self.createInitialNumpy(),
                           "PIXEL_TN": self.createInitialNumpy(), "PIXEL_FN": self.createInitialNumpy(),
                           "NUM_IMAGE": self.createInitialNumpy()}
        return initial_element

    def createInitialMetric(self):
        initial_metric = {"OBJECT_PRECISION": self.createInitialNumpy(), "OBJECT_RECALL": self.createInitialNumpy(),
                          "PIXEL_ACCURACY": self.createInitialNumpy(), "PIXEL_PRECISION": self.createInitialNumpy(),
                          "PIXEL_RECALL": self.createInitialNumpy(), "PIXEL_IOU": self.createInitialNumpy()}
        return initial_metric


    # CJY at 2020.10.17
    def update(self, saliency_maps, annotation, visual_labels, visualizer_name, layer_name, threshold=0.5):
        """
        :param saliency_maps: b, 1, w, h
        :param seg_gtmasks: b, c, w, h
        :param visual_labels: c
        :param visualizer_name:
        :param layer_name:
        :param threshold:
        :return:
        """
        # 1.生成二值化关注图&真值图
        if saliency_maps.shape[1] != 1:
            raise Exception("The shape[1] of saliency_map isn't 1.")

        binary_saliency_maps = torch.gt(saliency_maps, threshold)
        num_batch = saliency_maps.shape[0]

        # 2.
        for b in range(num_batch):
            key = "{}+{}+{}".format(visualizer_name, layer_name, threshold)
            if self.state.get(key) == None:
                self.state[key] = {"ELEMENT_OVERALL": self.createInitialElement(),
                                   "METRIC_OVERALL": self.createInitialMetric(),
                                   "METRIC_IMAGEWISE": self.createInitialMetric()}
            v_c_index = visual_labels[b]

            te = self.createInitialElement()
            tm = self.createInitialMetric()

            for s_c_index in range(self.num_seg_class):
                ann = annotation[b]

                # 计算mask 转numpy
                pt_mask = binary_saliency_maps[b][s_c_index].numpy().astype(np.uint8)   #.permute(1, 2, 0)
                gt_mask = binary_seg_gtmasks[b][s_c_index].numpy().astype(np.uint8)

                hit_mask = self.fillHoles(pt_mask, gt_mask)  # 保留的是二者相交的gt那部分
                miss_mask = gt_mask - hit_mask

                # 计算mask->scalar  通过联通域检测转化为标量
                hit_retval, hit_labels, hit_stats, hit_centroids = cv.connectedComponentsWithStats(hit_mask, connectivity=8, ltype=cv.CV_32S)
                object_hit = hit_retval - 1
                miss_retval, miss_labels, miss_stats, miss_centroids = cv.connectedComponentsWithStats(miss_mask, connectivity=8, ltype=cv.CV_32S)
                object_miss = miss_retval - 1

                pixel_tp = (pt_mask * gt_mask).sum(axis=(0, 1))
                pixel_fp = (pt_mask * (1 - gt_mask)).sum(axis=(0, 1))
                pixel_tn = ((1 - pt_mask) * (1 - gt_mask)).sum(axis=(0, 1))
                pixel_fn = ((1 - pt_mask) & gt_mask).sum(axis=(0, 1))

                te["OBJECT_HIT"][v_c_index][s_c_index] = object_hit
                te["OBJECT_MISS"][v_c_index][s_c_index] = object_miss
                te["PIXEL_TP"][v_c_index][s_c_index] = pixel_tp
                te["PIXEL_FP"][v_c_index][s_c_index] = pixel_fp
                te["PIXEL_TN"][v_c_index][s_c_index] = pixel_tn
                te["PIXEL_FN"][v_c_index][s_c_index] = pixel_fn
                te["NUM_IMAGE"][v_c_index][s_c_index] = 1

            tm["OBJECT_PRECISION"] = te["OBJECT_HIT"] / np.maximum(te["OBJECT_HIT"].sum(axis=1, keepdims=True), 1E-12)
            tm["OBJECT_RECALL"] = te["OBJECT_HIT"] / np.maximum((te["OBJECT_HIT"]+te["OBJECT_MISS"]), 1E-12)
            tm["PIXEL_ACCURACY"] = (te["PIXEL_TP"]+te["PIXEL_TN"])/np.maximum((te["PIXEL_TP"]+te["PIXEL_FP"]+te["PIXEL_TN"]+te["PIXEL_FN"]), 1E-12)
            tm["PIXEL_PRECISION"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FP"]), 1E-12)
            tm["PIXEL_RECALL"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FN"]), 1E-12)
            tm["PIXEL_IOU"] = te["PIXEL_TP"] / np.maximum((te["PIXEL_TP"] + te["PIXEL_FP"] + te["PIXEL_FN"]), 1E-12)

            # 记录
            for k in self.state[key]["METRIC_IMAGEWISE"].keys():
                self.state[key]["METRIC_IMAGEWISE"][k] = (self.state[key]["METRIC_IMAGEWISE"][k] * self.state[key]["ELEMENT_OVERALL"]["NUM_IMAGE"] + tm[k])\
                                                         /np.maximum((self.state[key]["ELEMENT_OVERALL"]["NUM_IMAGE"] + te["NUM_IMAGE"]), 1E-12)

            for k in self.state[key]["ELEMENT_OVERALL"].keys():
                self.state[key]["ELEMENT_OVERALL"][k] += te[k]

            self.state[key]["METRIC_OVERALL"]["OBJECT_PRECISION"] = self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] / np.maximum(self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"].sum(axis=1, keepdims=True), 1E-12)
            self.state[key]["METRIC_OVERALL"]["OBJECT_RECALL"] = self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["OBJECT_HIT"] + self.state[key]["ELEMENT_OVERALL"]["OBJECT_MISS"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_ACCURACY"] = (self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_TN"]) \
                                                                   / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_TN"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_PRECISION"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_RECALL"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)
            self.state[key]["METRIC_OVERALL"]["PIXEL_IOU"] = self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] / np.maximum((self.state[key]["ELEMENT_OVERALL"]["PIXEL_TP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FP"] + self.state[key]["ELEMENT_OVERALL"]["PIXEL_FN"]), 1E-12)


    def fillHoles(self, seedImg, Mask):
        # 输入是numpy格式
        seedImg = seedImg * Mask
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        i = 0
        while 1:
            dilated = cv.dilate(seedImg, kernel)  # 膨胀图像
            outputImg = dilated & Mask
            i = i + 1
            if i % 30 == 0:
                if (seedImg == outputImg).all():
                    break
                else:
                    seedImg = outputImg
            else:
                seedImg = outputImg
        return outputImg

    def saveXLS(self, savePath):
        # 1.生成总表，详细记录每一项
        sheetName = "SUMMARY-CLASSWISE"
        op_list = ["Visulization Method", "Observation Module", "Threshold",
                   "Metric Type", "Metric Name", "Visual Label", "Segmentation Label", "Value"]
        DF = pd.DataFrame(columns=op_list)

        for key in self.state:
            method_key, layer_key, th_key = key.split("+")
            for metric_type in self.state[key].keys():
                for metric_name in self.state[key][metric_type].keys():
                    metric_numpy = self.state[key][metric_type][metric_name]
                    for vi, vclass in enumerate(self.visual_class_list):
                        for si, sclass in enumerate(self.seg_class_list):
                            DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                        vclass, sclass, metric_numpy[vi][si]]], columns=op_list)
                            DF = pd.concat([DF, DF_NewLine], ignore_index=True)


        xls_filename = os.path.join(savePath, method_key + ".xlsx")
        """
        if os.path.exists(xls_filename) == True:
            with pd.ExcelWriter(xls_filename, mode='a') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        else:
            with pd.ExcelWriter(xls_filename, mode='w') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        """
        with pd.ExcelWriter(xls_filename, mode='w') as writer:
            DF.to_excel(writer, sheet_name=sheetName)

        #DF.to_excel(model.visualizer_name + ".xlsx")

        # 2.生成总表，详细记录每一项
        if self.visual_class_list == self.seg_class_list:
            sheetName = "SUMMARY"
            op_list = ["Visulization Method", "Observation Module", "Threshold",
                       "Metric Type", "Metric Name", "Visual Label", "Value"]
            DF = pd.DataFrame(columns=op_list)

            for key in self.state:
                method_key, layer_key, th_key = key.split("+")
                for metric_type in self.state[key].keys():
                    for metric_name in self.state[key][metric_type].keys():
                        metric_numpy = self.state[key][metric_type][metric_name]
                        mean_value = 0
                        for vi, vclass in enumerate(self.visual_class_list):
                            for si, sclass in enumerate(self.seg_class_list):
                                if vi == si:
                                    mean_value = mean_value + metric_numpy[vi][si]
                                    DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                                vclass, metric_numpy[vi][si]]], columns=op_list)
                                    DF = pd.concat([DF, DF_NewLine], ignore_index=True)

                        mean_value = mean_value / len(self.visual_class_list)
                        DF_NewLine = pd.DataFrame([[method_key, layer_key, th_key, metric_type, metric_name,
                                                    "mean", mean_value]], columns=op_list)
                        DF = pd.concat([DF, DF_NewLine], ignore_index=True)

            xls_filename = os.path.join(savePath, method_key + ".xlsx")
            """
            if os.path.exists(xls_filename) == True:
                with pd.ExcelWriter(xls_filename, mode='a') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            else:
                with pd.ExcelWriter(xls_filename, mode='w') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            """
            if os.path.exists(xls_filename) == True:
                with pd.ExcelWriter(xls_filename, mode='a') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)
            else:
                with pd.ExcelWriter(xls_filename, mode='w') as writer:
                    DF.to_excel(writer, sheet_name=sheetName)


        #2.记录统计后的表
        sheetName = "VISUAL-LABEL"
        op_list = ["Visulization Method", "Observation Module", "Threshold", "Visual Label",] + self.metric_name_list

        DF = pd.DataFrame(columns=op_list)

        for key in self.state:
            method_key, layer_key, th_key = key.split("+")
            for vi, vclass in enumerate(self.visual_class_list):
                line = [method_key, layer_key, th_key, vclass]
                for metric_name in self.metric_name_list:
                    metric_numpy = self.state[key]["METRIC_OVERALL"][metric_name]
                    line.append(metric_numpy[vi][vi])
                DF_NewLine = pd.DataFrame([line], columns=op_list)
                DF = pd.concat([DF, DF_NewLine], ignore_index=True)

        xls_filename = os.path.join(savePath, method_key + ".xlsx")
        if os.path.exists(xls_filename) == True:
            with pd.ExcelWriter(xls_filename, mode='a') as writer:
                DF.to_excel(writer, sheet_name=sheetName)
        else:
            with pd.ExcelWriter(xls_filename, mode='w') as writer:
                DF.to_excel(writer, sheet_name=sheetName)