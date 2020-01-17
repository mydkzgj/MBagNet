import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform

# 遮挡测试
"""
1.可以计算每一块而遮挡对logits或是score的影响（不同模型的关系）

2.可以计算遮挡一块与遮挡多块的非线性关系

"""


def generate_heatmap_pytorch(model, image, target, patchsize):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.

    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.

    """
    import torch

    with torch.no_grad():
        # pad with zeros
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize - 1) // 2:(patchsize - 1) // 2 + x, (patchsize - 1) // 2:(patchsize - 1) // 2 + y] = \
        image[0]
        image = padded_image[None].astype(np.float32)

        # turn to torch tensor
        input = torch.from_numpy(image).cuda()

        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 1000):
            global_feat, logits = model(batch_patches)
            logits = logits[:, target]  # [:, 0]
            logits_list.append(logits.data.cpu().numpy().copy())

        logits = np.hstack(logits_list)
        return logits.reshape((224, 224))