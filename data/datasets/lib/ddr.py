# encoding: utf-8
"""
@author:  cjy
@contact: sychenjiayang@163.com
"""

import glob
import re

import os.path as osp
import os

from data.datasets.bases import BaseImageDataset


class DDR_DR_GRADING(BaseImageDataset):
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/ddr_pre/dr_grading"

    def __init__(self, root='/home/cjy/data', verbose=True, **kwargs):
        super(DDR_DR_GRADING, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'valid')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train, train_statistics, train_c2l, train_l2c = self._process_dir(self.train_dir, relabel=True)
        val, val_statistics, val_c2l, val_l2c = self._process_dir(self.val_dir, relabel=False)
        test, test_statistics, test_c2l, test_l2c = self._process_dir(self.test_dir, relabel=False)

        if verbose:
            print("=> fundusTR loaded")
            print("train")
            print(train_statistics)
            print("val")
            print(val_statistics)
            print("test")
            print(test_statistics)
            #self.print_dataset_statistics(train, val, test)

        self.train = train
        self.val = val
        self.test = test

        self.num_categories = len(train_l2c)

        self.category = []
        for index in range(self.num_categories):
            self.category.append(train_l2c[index])

        self.category2label = train_c2l
        self.label2category = train_l2c

        self.num_train_statistics = train_statistics
        self.num_val_statistics = val_statistics
        self.num_test_statistics = test_statistics

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    #把图片名字都给解析了，其实医学图像也应该如此，不过先不管了
    def _process_dir(self, dir_path, relabel=False):
        #此数据集的标签存在txt文件中
        f = open(dir_path+'.txt')
        img2label = {}
        categorySet = set()
        categoryNum = {}
        dataset = []
        for line in f:
            img, label = line.split(" ")
            label = int(label)
            img2label[img] = label
            categorySet.add(label)
            if categoryNum.get(label) == None:
                categoryNum[label] = 1
            else:
                categoryNum[label] = categoryNum[label] + 1
            img_path = osp.join(dir_path, img)
            dataset.append((img_path, label))

        statistics = []

        category2label = {}
        label2category = {}
        for i in range(len(categorySet)):
            if i in categorySet:
                category2label[str(i)] = i
                label2category[i] = str(i)
                statistics.append((i, categoryNum[i]))
            else:
                raise Exception("Lack Category!", i)

        return dataset, statistics, category2label, label2category



class DDR_LESION_SEGMENTATION(BaseImageDataset):
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/ddr_pre/lesion_segmentation_regroup"#'ddr_DRgrading'

    def __init__(self, root='/home/cjy/data', verbose=True, **kwargs):
        super(DDR_LESION_SEGMENTATION, self).__init__()
        #self.dataset_grading_dir = osp.join(root, self.dataset_dir, "grading")
        self.dataset_segmentation_dir = osp.join(root, self.dataset_dir)

        #CJY 加入segmentation信息
        self.lesion = ["MA", "EX", "SE", "HE"]#["EX", "HE", "MA", "SE"]  #self.lesion = ["FUSION"]
        self.seg_train_dir = osp.join(self.dataset_segmentation_dir, 'train')
        self.seg_val_dir = osp.join(self.dataset_segmentation_dir, 'valid')
        self.seg_test_dir = osp.join(self.dataset_segmentation_dir, 'test')
        self.seg_train, seg_train_statistics, seg_train_c2l, seg_train_l2c = self._process_segmentation_dir(self.seg_train_dir)
        self.seg_val, seg_val_statistics, seg_val_c2l, seg_val_l2c = self._process_segmentation_dir(self.seg_val_dir)
        self.seg_test, seg_test_statistics, seg_test_c2l, seg_test_l2c = self._process_segmentation_dir(self.seg_test_dir)

        #self.num_categories = 4  #有4个等级的DR

        self.num_categories = len(seg_train_c2l)

        self.category = list(seg_train_c2l.keys())
        self.category.sort()

        self._check_before_run()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_segmentation_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_segmentation_dir))
        if not osp.exists(self.seg_train_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_train_dir))
        if not osp.exists(self.seg_val_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_val_dir))
        if not osp.exists(self.seg_test_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_test_dir))


    #CJY
    def _process_segmentation_dir(self, dir_path, relabel=False):
        f = open(dir_path + '.txt')
        categorySet = set()
        labelRecord = {}
        category2label = {}
        label2category = {}
        categoryNum = {}
        for line in f:
            img, label = line.split(" ")
            labelRecord[img] = int(label)
            categorySet.add(labelRecord[img])
            category2label[str(labelRecord[img])] = labelRecord[img]
            label2category[labelRecord[img]] = str(labelRecord[img])

            if categoryNum.get(labelRecord[img]) == None:
                categoryNum[labelRecord[img]] = 1
            else:
                categoryNum[labelRecord[img]] = categoryNum[labelRecord[img]] + 1

        statistics = []
        for c in categorySet:
            statistics.append((c, categoryNum[c]))

        dataset = []
        image_path = os.path.join(dir_path, "image")
        label_path = os.path.join(dir_path, "label")
        for imgfile in os.listdir(image_path):
            pre, ext = os.path.splitext(imgfile)
            if ext != ".JPG" and ext != ".jpg" :
                continue
            imagefullpath = os.path.join(image_path, imgfile)
            labelfullpathList = []
            for l in self.lesion:
                labelfullpath = os.path.join(label_path, l, imgfile.replace(".jpg", ".tif"))
                if os.path.exists(labelfullpath) == True:
                    labelfullpathList.append(labelfullpath)
                else:
                    raise Exception("Not find mask!")
            if os.path.exists(imagefullpath)==True:
                dataset.append((imagefullpath, labelfullpathList, labelRecord[imgfile]))

        return dataset, statistics, category2label, label2category


class DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION(BaseImageDataset):  # 用于弱监督，即返回病灶是否出现的多标签
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/ddr_pre/lesion_segmentation"  # 'ddr_DRgrading'

    def __init__(self, root='/home/cjy/data', verbose=True, **kwargs):
        super(DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION, self).__init__()
        # self.dataset_grading_dir = osp.join(root, self.dataset_dir, "grading")
        self.dataset_segmentation_dir = osp.join(root, self.dataset_dir)

        # CJY 加入segmentation信息
        self.lesion = ["MA", "EX", "SE", "HE"]  # ["EX", "HE", "MA", "SE"]  #self.lesion = ["FUSION"]
        self.seg_train_dir = osp.join(self.dataset_segmentation_dir, 'train')
        self.seg_val_dir = osp.join(self.dataset_segmentation_dir, 'valid')
        self.seg_test_dir = osp.join(self.dataset_segmentation_dir, 'test')
        self.seg_train, seg_train_statistics, seg_train_c2l, seg_train_l2c = self._process_segmentation_dir(
            self.seg_train_dir)
        self.seg_val, seg_val_statistics, seg_val_c2l, seg_val_l2c = self._process_segmentation_dir(self.seg_val_dir)
        self.seg_test, seg_test_statistics, seg_test_c2l, seg_test_l2c = self._process_segmentation_dir(
            self.seg_test_dir)

        # self.num_categories = 4  #有4个等级的DR

        self.num_categories = len(seg_train_c2l)

        self.category = list(seg_train_c2l.keys())
        self.category.sort()

        self._check_before_run()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_segmentation_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_segmentation_dir))
        if not osp.exists(self.seg_train_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_train_dir))
        if not osp.exists(self.seg_val_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_val_dir))
        if not osp.exists(self.seg_test_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_test_dir))

    # CJY
    def _process_segmentation_dir(self, dir_path, relabel=False):
        f = open(dir_path + '_lesion_multilabel.txt')
        labelRecord = {}
        category2label = {}
        label2category = {}
        categoryNum = {}

        category2label = {"MA":0, "EX":1, "SE":2, "HE":3}
        label2category = {0:"MA", 1:"EX", 2:"SE", 3:"HE"}

        for line in f:
            line = line.strip("\n")
            sub = line.split(" ")
            img = sub[0]
            label = [int(sub[i]) for i in range(1,5)]
            labelRecord[img] = label

        statistics = []

        dataset = []
        image_path = os.path.join(dir_path, "image")
        label_path = os.path.join(dir_path, "label")
        for imgfile in os.listdir(image_path):
            pre, ext = os.path.splitext(imgfile)
            if ext != ".JPG" and ext != ".jpg":
                continue
            imagefullpath = os.path.join(image_path, imgfile)
            labelfullpathList = []
            for l in self.lesion:
                labelfullpath = os.path.join(label_path, l, imgfile.replace(".jpg", ".tif"))
                if os.path.exists(labelfullpath) == True:
                    labelfullpathList.append(labelfullpath)
                else:
                    raise Exception("Not find mask!")
            if os.path.exists(imagefullpath) == True:
                dataset.append((imagefullpath, labelfullpathList, labelRecord[imgfile]))

        return dataset, statistics, category2label, label2category


class DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION_COLORMASK(BaseImageDataset):  # 用于弱监督，即返回病灶是否出现的多标签
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/ddr_pre/lesion_segmentation"  # 'ddr_DRgrading'

    def __init__(self, root='/home/cjy/data', verbose=True, **kwargs):
        super(DDR_LESION_SEGMENTATION_MULTILABEL_WEAKSURPERVISION_COLORMASK, self).__init__()
        # self.dataset_grading_dir = osp.join(root, self.dataset_dir, "grading")
        self.dataset_segmentation_dir = osp.join(root, self.dataset_dir)

        # CJY 加入segmentation信息
        self.lesion = ["MA", "EX", "SE", "HE"]  # ["EX", "HE", "MA", "SE"]  #self.lesion = ["FUSION"]
        self.seg_train_dir = osp.join(self.dataset_segmentation_dir, 'train')
        self.seg_val_dir = osp.join(self.dataset_segmentation_dir, 'valid')
        self.seg_test_dir = osp.join(self.dataset_segmentation_dir, 'test')
        self.seg_train, seg_train_statistics, seg_train_c2l, seg_train_l2c = self._process_segmentation_dir(
            self.seg_train_dir)
        self.seg_val, seg_val_statistics, seg_val_c2l, seg_val_l2c = self._process_segmentation_dir(self.seg_val_dir)
        self.seg_test, seg_test_statistics, seg_test_c2l, seg_test_l2c = self._process_segmentation_dir(
            self.seg_test_dir)

        # self.num_categories = 4  #有4个等级的DR

        self.num_categories = len(seg_train_c2l)

        self.category = list(seg_train_c2l.keys())
        self.category.sort()

        self._check_before_run()

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_segmentation_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_segmentation_dir))
        if not osp.exists(self.seg_train_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_train_dir))
        if not osp.exists(self.seg_val_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_val_dir))
        if not osp.exists(self.seg_test_dir):
            raise RuntimeError("'{}' is not available".format(self.seg_test_dir))

    # CJY
    def _process_segmentation_dir(self, dir_path, relabel=False):
        f = open(dir_path + '_lesion_multilabel_colormask_components_augumentation.txt')  #_lesion_multilabel_colormask.txt , _lesion_multilabel_colormask_components_augumentation.txt
        labelRecord = {}
        category2label = {}
        label2category = {}
        categoryNum = {}

        category2label = {"MA":0, "EX":1, "SE":2, "HE":3}
        label2category = {0:"MA", 1:"EX", 2:"SE", 3:"HE"}

        for line in f:
            line = line.strip("\n")
            sub = line.split(" ")
            img = sub[0]
            label = [int(sub[i]) for i in range(1,5)]
            labelRecord[img] = label

        statistics = []

        dataset = []
        image_path = os.path.join(dir_path, "color_mask_with_components_augumentation")  #color_mask, color_mask_with_single_label, color_mask_with_components_augumentation
        label_path = os.path.join(dir_path, "label")
        for imgfile in os.listdir(image_path):
            pre, ext = os.path.splitext(imgfile)
            if ext != ".JPG" and ext != ".jpg":
                continue
            imagefullpath = os.path.join(image_path, imgfile)
            labelfullpathList = []
            for l in self.lesion:
                labelfile = imgfile.split("-MA")[0].split("-EX")[0].split("-SE")[0].split("-HE")[0].split("_NONE")[0].split(".jpg")[0] + ".tif"
                labelfullpath = os.path.join(label_path, l, labelfile)
                if os.path.exists(labelfullpath) == True:
                    labelfullpathList.append(labelfullpath)
                else:
                    raise Exception("Not find mask!")
            if os.path.exists(imagefullpath) == True:
                dataset.append((imagefullpath, labelfullpathList, labelRecord[imgfile]))

        return dataset, statistics, category2label, label2category