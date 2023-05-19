import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from dataset import onehot2mask


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        # 针对每个batch都做一次loss计算，最后用mean作为本批次的loss进行反向传播
        batch_sizes = targets.size()[0]
        dice_eff = 0
        smooth = 1e-6
        # flat为batch_size, width * height的矩阵
        input_flat = input.view(batch_sizes, -1)
        targets_flat = targets.view(batch_sizes, -1)

        intersection = input_flat * targets_flat
        s_dice_coefficient = (2 * intersection.sum(1) + smooth) / \
            (input_flat.sum(1) + targets_flat.sum(1) + smooth)

        dice_eff += s_dice_coefficient.sum()
        # 计算一个批次中平均每张图的损失
        loss = 1 - (dice_eff / batch_sizes)
        return loss


class MultiClassDiceLoss(nn.Module):
    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, preds, labels):
        preds = F.softmax(preds, dim=1)
        preds = preds[:, 1:]  # 去掉背景类
        labels = labels[:, 1:]
        # 将预测和标签转换为一维向量
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        # 计算交集
        intersection = torch.sum(preds * labels)
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-8) / \
            (torch.sum(preds) + torch.sum(labels) + 1e-8)

        # 每个类别的平均 dice_loss
        return 1 - dice


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        eps = 1e-7
        loss_y1 = -1 * self.alpha * \
            torch.pow((1 - preds), self.gamma) * \
            torch.log(preds + eps) * labels
        loss_y0 = -1 * (1 - self.alpha) * torch.pow(preds,
                                                    self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_y0 + loss_y1
        return torch.mean(loss)


class MultiFocalLoss(nn.Module):
    def __init__(self):
        super(MultiFocalLoss, self).__init__()

    def forward(self, preds, labels):
        total_loss = 0
        binary_focal_loss = BinaryFocalLoss()
        logits = F.softmax(preds, dim=1)
        nums = labels.shape[1]
        for i in range(nums):
            loss = binary_focal_loss(logits[:, i], labels[:, i])
            total_loss += loss
        return total_loss / nums


class DiceCeLoss(nn.Module):
    def __init__(self):
        super(DiceCeLoss, self).__init__()

    def forward(self, preds, labels):
        assert preds.shape == labels.shape, "predict & target shape do not match"
        ce_loss = nn.CrossEntropyLoss()
        ce_total_loss = ce_loss(preds, labels)
        preds = F.softmax(preds, dim=1)

        # nums = labels.shape[1]

        preds = preds[:, 1:]  # 去掉背景类
        labels = labels[:, 1:]
        # 将预测和标签转换为一维向量
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)
        # 计算交集
        intersection = torch.sum(preds * labels)
        # 计算 Dice 系数
        dice = (2. * intersection + 1e-8) / \
            (torch.sum(preds) + torch.sum(labels) + 1e-8)

        if dice >= 1:
            dice = 1
        dice_ce_loss = -1 * torch.log(dice) + ce_total_loss
        # dice_ce_loss = 0.1 * (1 - dice) + 0.9 * ce_total_loss
        return dice_ce_loss
