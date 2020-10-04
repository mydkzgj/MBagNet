# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import cv2 as cv
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# CJY for color mask
def computeComponents(input):
    # tensor 转 numpt
    mask = input.gt(0).numpy().astype(np.uint8)

    # 联通域统计
    retval, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8, ltype=cv.CV_32S)

    return retval-1


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.only_obtain_label = False  # CJY at 2020.10.3 for sampler tranverse dataset rapidly

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, img_label = self.dataset[index]

        # CJY at 2020.10.3 for sampler tranverse dataset rapidly
        if self.only_obtain_label == True:
            return None, img_label, None

        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            img_label = self.target_transform(img_label)

        return img, img_label, img_path


# data.Dataset
# 所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
class SegmentationDataset(Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, dataset, transform=None, target_transform=None, seg_transforms=None, is_train=False):  # root表示图片路径
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.seg_transforms = seg_transforms
        self.is_train = is_train

        self.only_obtain_label = False  # CJY at 2020.10.3 for sampler tranverse dataset rapidly

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, mask_path, img_label = self.dataset[index]

        # CJY at 2020.10.3 for sampler tranverse dataset rapidly
        if self.only_obtain_label == True:
            return None, None, img_label, None

        # img
        img_pil = read_image(image_path)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = img_pil

        # target
        if isinstance(mask_path, str):
            mask_pil = Image.open(mask_path)
            if self.target_transform is not None:
                mask = self.target_transform(mask_pil)
            else:
                mask = mask_pil
        elif isinstance(mask_path, list):
            mask = []
            for mask_p in mask_path:
                mask_pil = Image.open(mask_p)
                if self.target_transform is not None:
                    mask.append(self.target_transform(mask_pil))              #[0:1]
                else:
                    mask.append(mask_pil)
        else:
            raise Exception("Wrong Mask Path Type")

        # both
        if self.seg_transforms is not None:
            img, mask = self.seg_transforms(img, mask)

        if isinstance(mask, list):
            mask = torch.cat(mask, dim=0)

        """
        if self.seg_num_classes == 1:
            mask = torch.max(mask, dim=0, keepdim=True)[0]
        elif self.seg_num_classes != mask.shape[0]:
            raise Exception("SEG_NUM_CLASSES cannot match the channels of mask")
        #"""

        return img, mask, img_label, image_path  # 返回的是图片



class DDRColormaskDataset(Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, dataset, transform=None, target_transform=None, seg_transforms=None, cfg=None, is_train=False):  # root表示图片路径
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.seg_transforms = seg_transforms
        self.is_train = is_train

        self.only_obtain_label = False            # CJY at 2020.10.3 for sampler tranverse dataset rapidly

        self.lesionTypeList = ["MA", "EX", "SE", "HE"]
        self.ColorMap = {
            "MA": torch.Tensor([0, 255, 0]).unsqueeze(1).unsqueeze(1) / 255.0,
            "EX": torch.Tensor([255, 255, 0]).unsqueeze(1).unsqueeze(1) / 255.0,
            "SE": torch.Tensor([255, 165, 0]).unsqueeze(1).unsqueeze(1) / 255.0,
            "HE": torch.Tensor([255, 0, 0]).unsqueeze(1).unsqueeze(1) / 255.0,
        }
        self.drawOrder = ["HE", "SE", "EX", "MA"]

        if self.is_train == True:
            self.shuffle_th = 0  # 0.5
            self.pick_channel_th = -1
        else:
            self.shuffle_th = 1
            self.pick_channel_th = 1

        self.norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, mask_path, img_label = self.dataset[index]

        # CJY at 2020.10.3 for sampler tranverse dataset rapidly
        if self.only_obtain_label == True:
            return None, None, img_label, None

        # img
        img_pil = read_image(image_path)
        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = img_pil

        # target  用img筛选mask
        mask = []
        origin_mask = []
        img_numpy = np.asarray(img_pil)
        img_numpy_nonzero = np.sum(img_numpy, axis=2) > 0
        for mask_p in mask_path:
            mask_pil = Image.open(mask_p)
            mask_numpy = np.asarray(mask_pil)
            mask_numpy = mask_numpy * img_numpy_nonzero
            mask_pil = Image.fromarray(mask_numpy)
            mk = self.target_transform(mask_pil)
            mask.append(mk.gt(0).float())
            o_mk = TF.to_tensor(TF.resize(mask_pil, (mk.shape[1], mk.shape[1]), Image.BOX))
            origin_mask.append(o_mk.gt(0).float())

        # both
        #if self.seg_transforms is not None:
        #    img, mask = self.seg_transforms(img, mask)

        if isinstance(mask, list):
            mask = torch.cat(mask, dim=0)

        # CJY at 2020.9.2  将mask按照draw order进行重叠位置得筛除（保留后来的，使得没有重叠位置）
        mask_no_overlap = mask.clone()
        _, imgName = os.path.split(image_path)
        if "HE" in imgName or "SE" in imgName or "EX" in imgName or "MA" in imgName:
            for lesionType in self.drawOrder:
                index = self.lesionTypeList.index(lesionType)
                if lesionType in imgName:
                    mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                    mask_no_overlap[index:index + 1] = mask[index:index + 1]
                else:
                    mask_no_overlap[index:index + 1] = mask_no_overlap[index:index + 1] * 0
        else:
            for lesionType in self.drawOrder:
                index = self.lesionTypeList.index(lesionType)
                mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                mask_no_overlap[index:index + 1] = mask[index:index + 1]

        # 将通道随机打乱
        if random.random() > self.shuffle_th:
            mask_no_overlap_list = [mask_no_overlap[i:i + 1] for i in range(mask_no_overlap.shape[0])]
            random.shuffle(mask_no_overlap_list)
            mask_no_overlap = torch.cat(mask_no_overlap_list, dim=0)

        # 依照mask_no_overlap进行绘制
        canvas = torch.zeros_like(img)
        for index, lesionType in enumerate(self.lesionTypeList):
            # sum = mask_no_overlap[index].sum()
            # sum = mask_no_overlap[index].sum().gt(0).int()
            # sum = torch.nn.functional.adaptive_max_pool2d(mask_no_overlap[index:index + 1], (7, 7)).sum()  #计算7*7max-sum
            sum = computeComponents(mask_no_overlap[index])  # 计算联通域个数
            img_label[index] = sum  # 用于回归
            if sum != 0:
                canvas = canvas * (1 - mask_no_overlap[index:index + 1]) + mask_no_overlap[index:index + 1] * self.ColorMap[self.lesionTypeList[index]]
            sum_mask = torch.sum(mask_no_overlap, dim=0, keepdim=True).ne(0).float()

        img = self.norm_transform(canvas) * sum_mask
        mask = torch.cat(origin_mask, dim=0)

        return img, mask, img_label, image_path  # 返回的是图片



class ColormaskDataset0000(Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, dataset, transform=None, target_transform=None, cfg=None, is_train=False):  # root表示图片路径
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        self.is_train = is_train

        self.only_obtain_label = False  # CJY at 2020.10.3 for sampler tranverse dataset rapidly

        self.ratio = cfg.DATA.TRANSFORM.MASK_SIZE_RATIO
        self.padding = cfg.DATA.TRANSFORM.PADDING   #不引入padding和crop
        if self.is_train == True:
            self.pad_num = self.padding * 2 - 1  #为了后面直接使用 加入 *2-1， 用于生成随机数
        else:
            self.pad_num = 0
        self.resizeH = cfg.DATA.TRANSFORM.SIZE[0]
        self.resizeW = cfg.DATA.TRANSFORM.SIZE[1]
        self.mask_resizeH = self.resizeH // self.ratio
        self.mask_resizeW = self.resizeW // self.ratio
        self.MaxPool = torch.nn.AdaptiveMaxPool2d((self.mask_resizeH, self.mask_resizeW))
        self.MaskPad = torch.nn.ZeroPad2d(self.padding//self.ratio)   #padding最好是ratio的倍数

        # 用于生成mask的维度
        self.seg_num_classes = cfg.MODEL.SEG_NUM_CLASSES

        # CJY at 2020.8.12
        self.generateColormask = 1
        if self.generateColormask == True:
            import torchvision.transforms as T
            normalize_transform = T.Normalize(mean=cfg.DATA.TRANSFORM.PIXEL_MEAN, std=cfg.DATA.TRANSFORM.PIXEL_STD)
            self.norm_transform = T.Compose([normalize_transform])

            from ..transforms.cla_transforms import RandomErasing
            if self.is_train == True:
                m_pad = cfg.DATA.TRANSFORM.PADDING * 4
                prob = cfg.DATA.TRANSFORM.PROB
                re_prob = cfg.DATA.TRANSFORM.RE_PROB
                self.shuffle_th = 0 #0.5
                self.pick_channel_th = -1
            else:
                m_pad = 0
                prob = 0
                re_prob = 0
                self.shuffle_th = 1
                self.pick_channel_th = 1

            self.single_mask_transform = T.Compose([
                # T.Resize(cfg.DATA.TRANSFORM.SIZE),
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=prob),
                T.Pad(m_pad),  # 暂时先去掉padding，因为有可能让mask中的病灶全部被剪切去
                T.RandomCrop(cfg.DATA.TRANSFORM.SIZE),
                T.ToTensor(),
                RandomErasing(probability=re_prob, sl=0.2, sh=0.5, mean=[0])
            ])

            self.to_tensor_transform = T.Compose([
                T.ToTensor(),
            ])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, mask_path_list, img_label = self.dataset[index]

        # CJY at 2020.10.3 for sampler tranverse dataset rapidly
        if self.only_obtain_label == True:
            return None, None, img_label, None

        img_pil = read_image(image_path)

        if self.transform is not None:
            img = self.transform(img_pil)

        mask_list = []
        for mask_path in mask_path_list:
            mask = Image.open(mask_path)
            if self.target_transform is not None:
                mask = self.target_transform(mask)
            mask_list.append(mask[0:1])    # cjy 由于读进来的可能是3通道，所以增加[0:1]
        mask = torch.cat(mask_list)

        # CJY at 2020.9.14 color_mask_with_components_augumentation 用img筛选mask
        if self.generateColormask == True:
            img_nonzero = torch.sum(self.to_tensor_transform(img_pil), dim=0, keepdim=True).gt(0).float()
            mask = mask * img_nonzero

        #上面让mask读入的为原图尺寸标签
        mask = self.MaxPool(mask)  #mask的resize选择maxpool

        # 随机剪切（不过像素级标签剪切平移是不是没什么用）  CJY 2020.8.3 目前处于关闭状态，也就是说没有数据扩增
        #"""
        if self.pad_num > 0:
            mask = self.MaskPad(mask)
            randH = random.randint(0, self.pad_num)
            randW = random.randint(0, self.pad_num)
        else:
            randH = 0
            randW = 0
        mask_randH = randH//self.ratio
        mask_randW = randW//self.ratio
        img = img[:, randH:randH + self.resizeH, randW:randW + self.resizeW]
        mask = mask[:, mask_randH:mask_randH + self.mask_resizeH, mask_randW:mask_randW + self.mask_resizeW]
        #"""

        if self.seg_num_classes == 1:
            mask = torch.max(mask, dim=0, keepdim=True)[0]
        elif self.seg_num_classes != mask.shape[0]:
            raise Exception("SEG_NUM_CLASSES cannot match the channels of mask")


        # CJY at 2020.8.12 在dataloader中生成colormask
        if self.generateColormask == True:
            # CJY at 2020.9.2
            # 将每一维得mask进行单独扩增
            mask_copy = mask.clone()
            for i in range(mask.shape[0]):
                mask[i:i+1] = self.single_mask_transform(mask[i:i+1])

            import os
            _, imgName = os.path.split(image_path)
            lesionTypeList = ["MA", "EX", "SE", "HE"]
            ColorMap = {
                "MA": torch.Tensor([0,255,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "EX": torch.Tensor([255,255,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "SE": torch.Tensor([255,165,0]).unsqueeze(1).unsqueeze(1)/255.0,
                "HE": torch.Tensor([255,0,0]).unsqueeze(1).unsqueeze(1)/255.0,
            }

            drawOrder = ["HE", "SE", "EX", "MA"]
            canvas = torch.zeros_like(img)

            # CJY at 2020.9.2  将mask按照draw order进行重叠位置得筛除（保留后来的，使得没有重叠位置）
            mask_no_overlap = mask.clone()
            if "HE" in imgName or "SE" in imgName or "EX" in imgName or "MA" in imgName:
                for lesionType in drawOrder:
                    index = lesionTypeList.index(lesionType)
                    if lesionType in imgName:
                        mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                        mask_no_overlap[index:index + 1] = mask[index:index + 1]
                    else:
                        mask_no_overlap[index:index + 1] = mask_no_overlap[index:index + 1] * 0
            else:
                for lesionType in drawOrder:
                    index = lesionTypeList.index(lesionType)
                    mask_no_overlap = mask_no_overlap * (1 - mask[index:index + 1])
                    mask_no_overlap[index:index + 1] = mask[index:index + 1]


            # 将通道随机打乱
            if random.random() > self.shuffle_th:
                mask_no_overlap_list = [mask_no_overlap[i:i + 1] for i in range(mask_no_overlap.shape[0])]
                random.shuffle(mask_no_overlap_list)
                mask_no_overlap = torch.cat(mask_no_overlap_list, dim=0)

            # 依照mask_no_overlap进行绘制
            for index, lesionType in enumerate(lesionTypeList):
                #sum = mask_no_overlap[index].sum()
                #sum = mask_no_overlap[index].sum().gt(0).int()
                #sum = torch.nn.functional.adaptive_max_pool2d(mask_no_overlap[index:index + 1], (7, 7)).sum()  #计算7*7max-sum
                sum = computeComponents(mask_no_overlap[index])    #计算联通域个数
                img_label[index] = sum  # 用于回归

                if sum != 0:
                    canvas = canvas * (1 - mask_no_overlap[index:index + 1]) + mask_no_overlap[index:index + 1] * ColorMap[lesionTypeList[index]]
                sum_mask = torch.sum(mask_no_overlap, dim=0, keepdim=True).ne(0).float()

            img = self.norm_transform(canvas) * sum_mask
            mask = mask_copy

        return img, mask, img_label, image_path  # 返回的是图片


