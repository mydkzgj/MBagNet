# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
"""

import random
from collections import defaultdict

from torch.utils.data.sampler import WeightedRandomSampler


def converMultiLabel2SingleLabel(multilabel, convertType="random"):
    if convertType == "random":
        # for random
        i_list = []
        for i, l in enumerate(multilabel):
            if l > 0:
                i_list.append(i)
        if i_list != []:
            int_label = random.choice(i_list)
        else:
            int_label = -1
    elif convertType == "max_random":
        # for max random  PACSCAL
        max_l = 0
        i_list = []
        for i, l in enumerate(multilabel):
            if l > max_l:
                max_l = l
                i_list.clear()
                i_list.append(i)
            elif l == max_l:
                i_list.append(i)
        if i_list != [] and max_l != 0:
            int_label = random.choice(i_list)
        else:
            int_label = -1
    elif convertType == "decimalism":
        # 十进制
        int_label = 0
        for l in multilabel:
            if l > 0:
                int_label = int_label * 10 + 1
            else:
                int_label = int_label * 10 + 0
    elif convertType == "binary":
        # 二进制
        int_label = 0
        for l in multilabel:
            if l > 0:
                int_label = int_label * 2 + 1
            else:
                int_label = int_label * 2 + 0
    else:
        raise Exception("Wrong ConvertType!")

    return int_label


#CJY at 2019.9.26
class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(self, data_source, max_num_categories, replacement=True):
        self.data_source = data_source

        if hasattr(self.data_source, "only_obtain_label") == True:
            self.data_source.only_obtain_label = True

        # 将data_source中的samples依照类别将同类的sample以列表的形式存入字典中
        self.index_dic = defaultdict(list)  # 这种字典与普通字典的却别？
        # for single-label and multi-label  at 2020.9.15
        num_tuple = len(self.data_source[0])
        label_index = 1 if num_tuple <= 3 else 2
        # for index, (_, label) in enumerate(self.data_source):
        for index, data in enumerate(self.data_source):  # 避免出现数据集返回值长度不一的情况
            label = data[label_index]
            if isinstance(label, int) == True:
                self.index_dic[label].append(index)
            elif isinstance(label, list) == True and isinstance(label[0], int)==True:
                if max_num_categories <= 6:  # 6的全组合数  2^6 = 64, 全组合数太多的话实在是分类负担，只能选择random选择其中一维类别
                    int_label = converMultiLabel2SingleLabel(label, convertType="decimalism")
                else:
                    int_label = converMultiLabel2SingleLabel(label, convertType="max_random")
                self.index_dic[int_label].append(index)
            else:
                raise Exception("Wrong Label Type")
        self.categories = list(self.index_dic.keys())

        if hasattr(self.data_source, "only_obtain_label") == True:
            self.data_source.only_obtain_label = False

        self.categories_weights_dict = {}
        self.num_samples = len(self.data_source)
        self.ratio = 1
        self.length = self.num_samples * self.ratio
        self.weights = [0] * self.num_samples
        for key in self.index_dic.keys():
            self.categories_weights_dict[key] = self.num_samples/len(self.index_dic[key])
            for i in self.index_dic[key]:
                self.weights[i] = self.categories_weights_dict[key]

        super(AutoWeightedRandomSampler, self).__init__(self.weights, self.num_samples)

    def __len__(self):
        return self.length
