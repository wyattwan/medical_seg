import numpy as np
import torch
import random
import torchvision.transforms as transforms
from PIL import Image


def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(
            x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(
            norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(
            norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transform_train):
        # img_ = np.array(img_) * 255
        img_ = img_.detach().numpy() * 255

    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception(
            "Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))

    return img_


def set_seed(seed):
    """
    进行随机种子的设置
    :param seed: 种子数
    :return: 无
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def rand_crop(data, label, img_w, img_h):
    width1 = random.randint(0, data.size[0] - img_w)
    height1 = random.randint(0, data.size[1] - img_h)
    width2 = width1 + img_w
    height2 = height1 + img_h

    data = data.crop((width1, height1, width2, height2))
    label = label.crop((width1, height1, width2, height2))

    return data, label


def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([0, 0, 0])
    cmap[1, :] = np.array([205, 85, 85])     # indianred
    cmap[2, :] = np.array([255, 182, 193])   # lightpink
    cmap[3, :] = np.array([255, 215, 0])     # gold
    cmap[4, :] = np.array([0, 255, 0])       # green
    cmap[5, :] = np.array([0, 255, 255])     # cyan
    cmap[6, :] = np.array([135, 206, 235])   # skyblue
    cmap[7, :] = np.array([230, 230, 250])   # lavender
    cmap[8, :] = np.array([255, 228, 196])   # bisque
    cmap[9, :] = np.array([255, 250, 250])   # snow

    return cmap
