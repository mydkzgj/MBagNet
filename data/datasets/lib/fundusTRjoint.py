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


class FundusTR_DRgrading(BaseImageDataset):
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/fundusTR-joint"#'ddr_DRgrading'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(FundusTR_DRgrading, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'valid')
        self.test_dir = osp.join(self.dataset_dir, 'test')

        #self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        #self.query_dir = osp.join(self.dataset_dir, 'query')
        #self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, train_statistics, train_c2l, train_l2c = self._process_dir(self.train_dir, relabel=True)
        val, val_statistics, val_c2l, val_l2c = self._process_dir(self.val_dir, relabel=False)
        test, test_statistics, test_c2l, test_l2c = self._process_dir(self.test_dir, relabel=False)

        #train = self._process_dir(self.train_dir, relabel=True)
        #query = self._process_dir(self.query_dir, relabel=False)
        #gallery = self._process_dir(self.gallery_dir, relabel=False)

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

        """
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        """

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
        categroyList = os.listdir(dir_path)
        dataset = []
        statistics = []
        category2label = {}
        label2category = {}

        for index, category in enumerate(categroyList):
            category2label[category] = index
            label2category[index] = category
            if "." in category:
                continue
            category_path = osp.join(dir_path, category, "img")   #CJY at 2020.7.11
            c_i = 0
            imgList = os.listdir(category_path)
            for img in imgList:
                img_path = osp.join(category_path, img)
                dataset.append((img_path, index))
                c_i = c_i + 1
            statistics.append((category, c_i))

        return dataset, statistics, category2label, label2category



class FundusTR_DRgrading_WeakSupervision(BaseImageDataset):   #用于弱监督
    """
    DDR_DRgrading
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = "DATABASE/fundusTR-joint"#'ddr_DRgrading'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(FundusTR_DRgrading_WeakSupervision, self).__init__()
        #self.dataset_grading_dir = osp.join(root, self.dataset_dir, "grading")
        self.dataset_segmentation_dir = osp.join(root, self.dataset_dir)


        #CJY 加入segmentation信息
        self.lesion = ["MA", "EX", "SE", "HE", "ML"]#["EX", "HE", "MA", "SE"]  #self.lesion = ["FUSION"]
        self.seg_train_dir = osp.join(self.dataset_segmentation_dir, 'train')
        self.seg_val_dir = osp.join(self.dataset_segmentation_dir, 'valid')
        self.seg_test_dir = osp.join(self.dataset_segmentation_dir, 'test')
        self.seg_train, self.train_c2l, self.train_l2c = self._process_segmentation_dir(self.seg_train_dir)
        self.seg_val, self.val_c2l, self.val_l2c = self._process_segmentation_dir(self.seg_val_dir)
        self.seg_test, self.test_c2l, self.test_l2c = self._process_segmentation_dir(self.seg_test_dir)

        self.num_categories = 2  # stable worse

        self.category = []
        for index in range(self.num_categories):
            self.category.append(self.train_l2c[index])

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
        dataset = []
        category2label = {}
        label2category = {}
        categroyList = os.listdir(dir_path)
        for index, category in enumerate(categroyList):
            category2label[category] = index
            label2category[index] = category
            if "." in category:
                continue
            category_path = osp.join(dir_path, category)
            image_path = os.path.join(category_path, "img")
            label_path = os.path.join(category_path, "mask/category")
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
                    dataset.append((imagefullpath, labelfullpathList, index))

        return dataset, category2label, label2category