# encoding: utf-8
"""
@author:  cjy
@contact: sychenjiayang@163.com
"""
import os
import tarfile
import collections
from torchvision.datasets.vision import VisionDataset
import xml.etree.ElementTree as ET
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg



class CocoClassification(VisionDataset):
    """
    CJY at 2020.9.29
    VOCClassification
    torchvision: offer detection & segmentation
    costum classification multilabel
    """
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super(CocoClassification, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.num_classes = len(self.coco.dataset["categories"])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        label = self.createMultiLabel(target)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label  #target

    def __len__(self):
        return len(self.ids)

    def createMultiLabel(self, target):
        multi_label = [0] * self.num_classes
        for t in target:
            category_id = t["category_id"]
            multi_label[category_id] = multi_label[category_id] + 1
            print(category_id)

        return multi_label