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

import numpy as np

DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': os.path.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': os.path.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    },
    '2007-test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }
}


class VOCClassification(VisionDataset):
    """
    CJY at 2020.9.29
    VOCClassification
    torchvision: offer detection & segmentation
    costum classification multilabel
    """
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years 2007 to 2012.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
                (default: alphabetic indexing of VOC's 20 classes).
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
        """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None):
        super(VOCClassification, self).__init__(root, transforms, transform, target_transform)
        self.year = year
        if year == "2007" and image_set == "test":
            year = "2007-test"
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007-test":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        #with open(os.path.join(split_f), "r") as f:
        #    file_names = [x.strip() for x in f.readlines()]
        #self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        #self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]

        self.__init_classes()
        self.images, self.annotations, self.labels = self.__dataset_info(record_txt=split_f, image_dir=image_dir, annotation_dir=annotation_dir)

        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        label = self.labels[index]
        if self.transform !=None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def __dataset_info(self, record_txt, image_dir, annotation_dir):
        with open(record_txt) as f:
            filenames = f.readlines()
        filenames = [x.strip() for x in filenames]

        images = []
        annotations = []
        labels = []
        for filename in filenames:
            # image path
            imagefilename = os.path.join(image_dir, filename + '.jpg')
            images.append(imagefilename)
            # xml path
            xmlfilename = os.path.join(annotation_dir, filename + '.xml')
            annotations.append(xmlfilename)
            # label
            tree = ET.parse(xmlfilename)
            objs = tree.findall('object')
            multi_label = [0] * self.num_classes
            for ix, obj in enumerate(objs):
                obj_class_index = self.class_to_ind[obj.find('name').text.lower().strip()]
                #multi_label[obj_class_index] = 1
                multi_label[obj_class_index] = multi_label[obj_class_index] + 1
            labels.append(multi_label)

        return images, annotations, labels

    def __init_classes(self):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)