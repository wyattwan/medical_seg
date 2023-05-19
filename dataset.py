import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def make_dataset(root):
    # root = "./dataset/train/"
    imgs = []    # 空列表用于存放原图和mask图
    ori_path = os.path.join(root, "png_origin")
    ground_path = os.path.join(root, "png_mask")
    names = os.listdir(ori_path)
    names.sort(key=lambda x: int(x.split('.')[0]))
    n = len(names)
    for i in range(n):
        img = os.path.join(ori_path, names[i])
        mask = os.path.join(ground_path, names[i])
        imgs.append((img, mask))
    return imgs


class MSDataset(Dataset):   # 上颚窦分割类
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('L')   # 转为灰度
        img_y = Image.open(y_path).convert('L')
        img_y = np.asarray(img_y)
        img_y = torch.tensor(mask2onehot(img_y, 10), dtype=torch.float)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
