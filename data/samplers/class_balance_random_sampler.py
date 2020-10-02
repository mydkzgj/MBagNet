# encoding: utf-8
"""
@author:  JiayangChen
@contact: sychenjiayang@163.com
origin: triplet_sampler
"""

import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler


#CJY at 2019.9.26
class ClassBalanceRandomSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, label).
    - num_categories_per_batch (int): number of categories per batch.
    - num_instances_per_category (int): number of instances per category in a batch.
    - batch_size (int): number of examples in a batch.
    -
    """

    def __init__(self, data_source, num_categories_per_batch, num_instances_per_category, max_num_categories, is_train=True):
        self.data_source = data_source
        self.num_categories_per_batch = num_categories_per_batch
        self.num_instances_per_category = num_instances_per_category
        self.batch = num_categories_per_batch * num_instances_per_category
        self.is_train = is_train

        #num_categories_per_batch不能小于总的类别数
        if self.num_categories_per_batch > max_num_categories:# or self.num_categories_per_batch < 2:
            raise Exception("Invalid Num_categories_per_batch!", self.num_categories_per_batch)

        #将data_source中的samples依照类别将同类的sample以列表的形式存入字典中
        self.index_dic = defaultdict(list)  #这种字典与普通字典的却别？
        # for single-label and multi-label  at 2020.9.15
        for index, (_, label) in enumerate(self.data_source):
            if isinstance(label, int)==True:
                self.index_dic[label].append(index)
            elif isinstance(label, list)==True:
                """
                # for random
                i_list = []
                for i, l in enumerate(label):
                    if l > 0:
                        i_list.append(i)
                if i_list != []:
                    int_label = random.choice(i_list)
                else:
                    int_label = -1                    
                """
                #"""
                # for max random  PACSCAL
                max = 0
                i_list = []
                for i, l in enumerate(label):
                    if l > max:
                        max = l
                        i_list.clear()
                        i_list.append(i)
                    elif l == max:
                        i_list.append(i)
                if i_list != []:
                    int_label = random.choice(i_list)
                else:
                    int_label = -1
                #"""
                """
                int_label = 0
                for l in label:
                    if l > 0:
                        int_label = int_label * 10 + 1
                    else:
                        int_label = int_label * 10 + 0
                #"""
                self.index_dic[int_label].append(index)

        self.categories = list(self.index_dic.keys())

        #记录每类的sample数量，并找出最大sample数量的类别（用于后续平衡其他类别的标准）
        self.targetNum_instances_per_category = {}
        max_num_samples = 0
        min_num_samples = 100000000
        for category in self.index_dic.keys():
            self.targetNum_instances_per_category[category] = len(self.index_dic[category])
            if max_num_samples < self.targetNum_instances_per_category[category]:
                max_num_samples =  self.targetNum_instances_per_category[category]
            if min_num_samples > self.targetNum_instances_per_category[category]:
                min_num_samples =  self.targetNum_instances_per_category[category]
        if max_num_samples % self.num_instances_per_category == 0:  #保证每类的samples含有整倍数的instances
            self.max_num_samples = max_num_samples
        else:
            self.max_num_samples = (max_num_samples//self.num_instances_per_category + 1) * self.num_instances_per_category
        #self.max_num_samples = 50 * self.num_instances_per_category   #减少训练集样本数量，加快训练速度
        self.min_num_samples = min_num_samples

        #设置每类的样本需要构建数量
        if self.is_train == True:
            for category in self.index_dic.keys():
                self.targetNum_instances_per_category[category] = self.max_num_samples
        else:
            pass
            #为了加快验证时间所做，后期移除
            #for category in self.index_dic.keys():
            #    self.targetNum_instances_per_category[category] = self.min_num_samples

        #epoch内样本数量
        self.length = 0
        for category in self.index_dic.keys():
            self.length = self.length + self.targetNum_instances_per_category[category]


    def __iter__(self):  #核心函数，返回一个迭代器
        # 将instances按每self.num_instances_per_category为一组存储到类别索引的字典里
        batch_idxs_dict = defaultdict(list)

        for category in self.categories:
            # 1.首先通过循环串联每类的样本（乱序）来增加该类的待选样本长度
            idxs = []
            while len(idxs) < self.targetNum_instances_per_category[category]:
                copy_idxs = copy.deepcopy(self.index_dic[category])
                random.shuffle(copy_idxs)   #为了测试attention而注掉
                idxs = idxs + copy_idxs
            idxs = idxs[0 : self.targetNum_instances_per_category[category]]

            #2.将每类样本按每self.num_instances_per_category一组分割（即一个batch内的instance）
            batch_idxs = []
            for i, idx in enumerate(idxs):
                batch_idxs.append(idx)
                if i == self.targetNum_instances_per_category[category]-1:  #不舍弃后续不足一个batch内instance—per-category的样本（非训练情形）
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []
                    break
                if len(batch_idxs) == self.num_instances_per_category:
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []

        #随机抽取组成batch，即设定迭代计划
        copy_categories = copy.deepcopy(self.categories)
        final_idxs = []   #迭代器核心列表
        if self.is_train == True:
            num_categories_th = self.num_categories_per_batch - 1
            while len(copy_categories) > num_categories_th:   #若其小于每个batch需要抽取的class则停止
                if self.is_train == True:
                    selected_categories = random.sample(copy_categories, self.num_categories_per_batch)   #随机挑选类别

                batch_idxs = []
                for category in selected_categories:
                    batch_idxs += batch_idxs_dict[category].pop(0)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
                #batch_idxs1 = batch_idxs[0:len(batch_idxs)//4]
                #batch_idxs2 = batch_idxs[len(batch_idxs)//4:len(batch_idxs)]
                #random.shuffle(batch_idxs2)
                #batch_idxs = batch_idxs1 + batch_idxs2
                final_idxs.extend(batch_idxs)
        else:
            num_categories_th = 0
            while len(copy_categories) > num_categories_th:  # 若其小于每个batch需要抽取的class则停止
                if len(copy_categories) >= self.num_categories_per_batch:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=False)  # 随机挑选类别
                    #selected_categories = np.array([0,1,2,3,4,5])   #为了测试attention加入
                else:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=True)  # 随机挑选类别

                for category in selected_categories:
                    if len(batch_idxs_dict[category]) <= 0:
                        continue
                    batch_idxs = batch_idxs_dict[category].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
        self.epoch_num_samples = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length



#CJY at 2019.9.26
class ClassBalanceRandomSamplerForSegmentation(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, label).
    - num_categories_per_batch (int): number of categories per batch.
    - num_instances_per_category (int): number of instances per category in a batch.
    - batch_size (int): number of examples in a batch.
    -
    """

    def __init__(self, data_source, num_categories_per_batch, num_instances_per_category, max_num_categories, is_train=True):
        self.data_source = data_source
        self.num_categories_per_batch = num_categories_per_batch
        self.num_instances_per_category = num_instances_per_category
        self.batch = num_categories_per_batch * num_instances_per_category
        self.is_train = is_train

        #将data_source中的samples依照类别将同类的sample以列表的形式存入字典中
        self.index_dic = defaultdict(list)  #这种字典与普通字典的却别？
        # for single-label and multi-label  at 2020.9.15
        for index, (_, _, label) in enumerate(self.data_source):
            if isinstance(label, int)==True:
                self.index_dic[label].append(index)
            elif isinstance(label, list)==True:
                int_label = 0
                for l in label:
                    if l > 0:
                        int_label = int_label * 10 + 1
                    else:
                        int_label = int_label * 10 + 0
                self.index_dic[int_label].append(index)
        self.categories = list(self.index_dic.keys())

        #记录每类的sample数量，并找出最大sample数量的类别（用于后续平衡其他类别的标准）
        self.targetNum_instances_per_category = {}
        max_num_samples = 0
        min_num_samples = 10000000
        for category in self.index_dic.keys():
            self.targetNum_instances_per_category[category] = len(self.index_dic[category])
            if max_num_samples < self.targetNum_instances_per_category[category]:
                max_num_samples =  self.targetNum_instances_per_category[category]
            if min_num_samples > self.targetNum_instances_per_category[category]:
                min_num_samples =  self.targetNum_instances_per_category[category]
        if max_num_samples % self.num_instances_per_category == 0:  #保证每类的samples含有整倍数的instances
            self.max_num_samples = max_num_samples
        else:
            self.max_num_samples = (max_num_samples//self.num_instances_per_category + 1) * self.num_instances_per_category
        #self.max_num_samples = 400 * self.num_instances_per_category   #减少训练集样本数量，加快训练速度
        self.min_num_samples = min_num_samples

        #设置每类的样本需要构建数量
        if self.is_train == True:
            for category in self.index_dic.keys():
                self.targetNum_instances_per_category[category] = self.max_num_samples
        else:
            pass
            #为了加快验证时间所做，后期移除
            #for category in self.index_dic.keys():
            #    self.targetNum_instances_per_category[category] = self.min_num_samples

        #epoch内样本数量
        self.length = 0
        for category in self.index_dic.keys():
            self.length = self.length + self.targetNum_instances_per_category[category]


    def __iter__(self):  #核心函数，返回一个迭代器
        # 将instances按每self.num_instances_per_category为一组存储到类别索引的字典里
        batch_idxs_dict = defaultdict(list)

        for category in self.categories:
            # 1.首先通过循环串联每类的样本（乱序）来增加该类的待选样本长度
            idxs = []
            while len(idxs) < self.targetNum_instances_per_category[category]:
                copy_idxs = copy.deepcopy(self.index_dic[category])
                random.shuffle(copy_idxs)   #为了测试attention而注掉
                idxs = idxs + copy_idxs
            idxs = idxs[0 : self.targetNum_instances_per_category[category]]

            #2.将每类样本按每self.num_instances_per_category一组分割（即一个batch内的instance）
            batch_idxs = []
            for i, idx in enumerate(idxs):
                batch_idxs.append(idx)
                if i == self.targetNum_instances_per_category[category]-1:  #不舍弃后续不足一个batch内instance—per-category的样本（非训练情形）
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []
                    break
                if len(batch_idxs) == self.num_instances_per_category:
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []

        #随机抽取组成batch，即设定迭代计划
        copy_categories = copy.deepcopy(self.categories)
        final_idxs = []   #迭代器核心列表
        if self.is_train == True:
            num_categories_th = self.num_categories_per_batch - 1
            while len(copy_categories) > num_categories_th:#2:#num_categories_th:   #若其小于每个batch需要抽取的class则停止
                if self.is_train == True:
                    sc = copy_categories.pop(0)
                    selected_categories = [sc]
                    copy_categories.append(sc)
                    #selected_categories = random.sample(copy_categories, self.num_categories_per_batch)   #随机挑选类别

                batch_idxs = []
                for category in selected_categories:
                    batch_idxs += batch_idxs_dict[category].pop(0)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
                #batch_idxs1 = batch_idxs[0:len(batch_idxs)//4]
                #batch_idxs2 = batch_idxs[len(batch_idxs)//4:len(batch_idxs)]
                #random.shuffle(batch_idxs2)
                #batch_idxs = batch_idxs1 + batch_idxs2
                final_idxs.extend(batch_idxs)
        else:
            num_categories_th = 0
            while len(copy_categories) > num_categories_th:  # 若其小于每个batch需要抽取的class则停止
                if len(copy_categories) >= self.num_categories_per_batch:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=False)  # 随机挑选类别
                    #selected_categories = np.array([0,1,2,3,4,5])   #为了测试attention加入
                else:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=True)  # 随机挑选类别

                for category in selected_categories:
                    if len(batch_idxs_dict[category]) <= 0:
                        continue
                    batch_idxs = batch_idxs_dict[category].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
        self.epoch_num_samples = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length