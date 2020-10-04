import torchvision
import os

from ..transforms import build_transforms, build_seg_transforms, build_transforms_for_colormask

from .lib import init_dataset
from .custom_dataset import ImageDataset, SegmentationDataset, DDRColormaskDataset

from .pascal_voc_classification import VOCClassification
from .coco_classification import CocoClassification

def make_dataset_for_custom_datasets(cfg, for_train):
    dataset = init_dataset(cfg.DATA.DATASETS.NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)
    classes_list = dataset.category  # 建立classes_list

    if "colormask" in cfg.DATA.DATASETS.NAMES:
        train_transforms, train_target_transforms = build_transforms_for_colormask(cfg, is_train=for_train)
        val_transforms, val_target_transforms = build_transforms_for_colormask(cfg, is_train=False)
        test_transforms, test_target_transforms = build_transforms_for_colormask(cfg, is_train=False)
        train_set = DDRColormaskDataset(dataset.train, train_transforms, train_target_transforms)
        val_set = DDRColormaskDataset(dataset.val, val_transforms, val_target_transforms)
        test_set = DDRColormaskDataset(dataset.test, test_transforms, test_target_transforms)
    else:
        train_transforms = build_transforms(cfg, is_train=for_train)
        val_transforms = build_transforms(cfg, is_train=False)
        test_transforms = build_transforms(cfg, is_train=False)
        train_set = ImageDataset(dataset.train, train_transforms)
        val_set = ImageDataset(dataset.val, val_transforms)
        test_set = ImageDataset(dataset.test, test_transforms)

    return train_set, val_set, test_set, classes_list

def make_seg_dataset_for_custom_datasets(cfg, for_train):
    dataset = init_dataset(cfg.DATA.DATASETS.SEG_NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)
    classes_list = dataset.category  # 建立classes_list

    if "colormask" in cfg.DATA.DATASETS.SEG_NAMES:
        train_transforms, train_target_transforms = build_transforms_for_colormask(cfg, is_train=for_train)
        val_transforms, val_target_transforms = build_transforms_for_colormask(cfg, is_train=False)
        test_transforms, test_target_transforms = build_transforms_for_colormask(cfg, is_train=False)
        train_set = DDRColormaskDataset(dataset.seg_train, train_transforms, train_target_transforms)
        val_set = DDRColormaskDataset(dataset.seg_val, val_transforms, val_target_transforms)
        test_set = DDRColormaskDataset(dataset.seg_test, test_transforms, test_target_transforms)
    else:
        train_seg_transforms = build_seg_transforms(cfg, is_train=for_train)
        val_seg_transforms = build_seg_transforms(cfg, is_train=False)
        test_seg_transforms = build_seg_transforms(cfg, is_train=False)
        train_set = SegmentationDataset(dataset.seg_train, seg_transforms=train_seg_transforms, is_train=for_train)
        val_set = SegmentationDataset(dataset.seg_val, seg_transforms=val_seg_transforms, is_train=False)
        test_set = SegmentationDataset(dataset.seg_test, seg_transforms=test_seg_transforms, is_train=False)

    return train_set, val_set, test_set, classes_list


def make_dataset_for_classic_datasets(cfg, for_train):
    train_transforms = build_transforms(cfg, is_train=for_train)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    root_path = cfg.DATA.DATASETS.ROOT_DIR

    # for those have been realized by torchvision
    if cfg.DATA.DATASETS.NAMES == "cifar10":
        root_path = os.path.join(root_path, "DATABASE", "CIFAR10")
        train_set = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "cifar100":
        root_path = os.path.join(root_path, "DATABASE", "CIFAR100")
        train_set = torchvision.datasets.CIFAR100(root=root_path, train=True, download=True, transform=train_transforms)
        val_set = torchvision.datasets.CIFAR100(root=root_path, train=False, download=True, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-classification":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = VOCClassification(root=root_path, year="2012", image_set="train", download=False, transform=train_transforms)
        val_set = VOCClassification(root=root_path, year="2012", image_set="val", download=False, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.classes
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-detection":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCDetection(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    elif cfg.DATA.DATASETS.NAMES == "pascal-voc-segmentation":
        root_path = os.path.join(root_path, "DATABASE", "PASCAL-VOC")
        train_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="train", download=True, transform=train_transforms)
        val_set = torchvision.datasets.VOCSegmentation(root=root_path, year="2012", image_set="val", download=True, transform=val_transforms)
        test_set = val_set
        classes_list = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    elif cfg.DATA.DATASETS.NAMES == "coco-classification":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "instances_{}2017.json".format("train"))
        train_set = CocoClassification(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "instances_{}2017.json".format("val"))
        val_set = CocoClassification(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = train_set.class_list
    elif cfg.DATA.DATASETS.NAMES == "coco-caption":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "captions_{}2017.json".format("train"))
        train_set = torchvision.datasets.CocoCaptions(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "captions_{}2017.json".format("val"))
        val_set = torchvision.datasets.CocoCaptions(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = None
    elif cfg.DATA.DATASETS.NAMES == "coco-detection":
        root_path = os.path.join(root_path, "DATABASE", "Microsoft-COCO")
        year = "2017"
        annotation_path = os.path.join(root_path, "annotations")
        train_image_path = os.path.join(root_path, "{}{}".format("train", year))
        train_annfile =os.path.join(annotation_path, "captions_{}{}.json".format("train", year))
        train_set = torchvision.datasets.CocoCaptions(root=train_image_path, annFile=train_annfile, transform=train_transforms)
        val_image_path = os.path.join(root_path, "{}{}".format("val", year))
        val_annfile = os.path.join(annotation_path, "captions_{}{}.json".format("val", year))
        val_set = torchvision.datasets.CocoCaptions(root=val_image_path, annFile=val_annfile, transform=val_transforms)
        test_set = val_set
        classes_list = [train_set.coco.dataset["categories"][i]["name"] for i in range(len(train_set.coco.dataset["categories"]))]

    return train_set, val_set, test_set, classes_list