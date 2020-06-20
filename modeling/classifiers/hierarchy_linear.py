import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyLinear(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, in_planes=None, num_classes=None):

        super(HierarchyLinear, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes

        self.classifier1 = nn.Linear(self.in_planes, 1)
        #self.classifier1.apply(weights_init_classifier)
        self.classifier2 = nn.Linear(self.in_planes, 1)
        #self.classifier2.apply(weights_init_classifier)
        self.classifier3 = nn.Linear(self.in_planes, 1)
        #self.classifier3.apply(weights_init_classifier)
        self.classifier4 = nn.Linear(self.in_planes, 1)
        #self.classifier4.apply(weights_init_classifier)
        self.classifier5 = nn.Linear(self.in_planes, 1)
        #self.classifier5.apply(weights_init_classifier)
        self.classifier6 = nn.Linear(self.in_planes, 1)
        #self.classifier6.apply(weights_init_classifier)


    def forward(self, x):
        # 注：对于2分类问题，不应该用softmax，因为classifier会有2个超平面，而实际上只需要一个超平面
        logits1 = self.classifier1(x)  # 画质（0，1，2，3，4，）Vs 5
        score1 = F.sigmoid(logits1)
        logits2 = self.classifier2(x)  # 正常 0， （1，2，3，4）
        score2 = F.sigmoid(logits2)
        logits3 = self.classifier3(x)  # 病灶 1, (2，3，4)
        score3 = F.sigmoid(logits3)
        logits4 = self.classifier4(x)  # 病灶 2, (3，4)
        score4 = F.sigmoid(logits4)
        logits5 = self.classifier5(x)  # 病灶 3, (4)
        score5 = F.sigmoid(logits5)

        score_c5 = score1
        score_L0 = 1 - score1
        score_c0 = score_L0 * score2
        score_L1 = score_L0 * (1 - score2)
        score_c1 = score_L1 * score3
        score_L2 = score_L1 * (1 - score3)
        score_c2 = score_L2 * score4
        score_L3 = score_L2 * (1 - score4)
        score_c3 = score_L3 * score5
        score_c4 = score_L3 * (1 - score5)

        final_logits = torch.cat([score_c0, score_c1, score_c2, score_c3, score_c4, score_c5], dim=1)
        # final_logits = torch.log(final_logits)  # 放到了loss里面，和nlll组合
        return final_logits