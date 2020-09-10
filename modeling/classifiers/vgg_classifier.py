import torch
import torch.nn as nn
import torch.nn.functional as F


class VggClassifier(nn.Module):
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

        super(VggClassifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes

        self.add_module(
            "main",  nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),  # True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(),  # True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes),
            )
        )

