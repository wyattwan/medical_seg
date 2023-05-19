import torch
from torch.cuda.amp import autocast
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from torch.autograd import Variable
from unet import TransUnet
from unet import Res1Unet
from unet import Res2Unet
from unet import TransRes1Unet
from unet import Unet
from dataset import MSDataset
from common_tools import transform_invert
from dataset import onehot2mask
from dataset import mask2onehot
import my_loss

val_interval = 1   # 验证步长

x_transforms = transforms.Compose([
    transforms.ToTensor(),
])
# y_transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])
# x_transforms = None
y_transforms = None

train_curve = list()
valid_curve = list()


def train_model(args, model, criterion, optimizer, dataload):
    epoch_loss_last = 1000   # 保存上次训练loss，方便判断best.pth
    epoch_valid_loss_last = 1000
    epoch_best_last = 10000
    if os.path.exists(args.ckpt):  # 模型保存路径
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
        start_epoch = 0
        print('加载模型成功！')
    else:
        start_epoch = 0
        print('无模型，从头开始训练')

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        print('-' * 10)
        print(' Epoch {}/{}'.format(epoch, args.epochs))
        dt_size = len(dataload.dataset)
        model.train(mode=True)  # 保留dropout 模块
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            with torch.autocast(device_type='cuda', dtype=torch.float32):
                step += 1
                inputs = x.to(args.device)
                labels = y.to(args.device)
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()   # 反向传播
            scaler.step(optimizer=optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()
            epoch_loss += loss.item()
            train_curve.append(loss.item())
            print("\r %d/%d,train_loss:%0.6f" % (step, (dt_size - 1) //
                  dataload.batch_size + 1, loss.item()), end='')
        print("\r epoch %d train_loss:%0.6f" %
              (epoch, epoch_loss/step), end='')
        print('\r\n')
        if epoch_loss <= epoch_loss_last:
            epoch_loss_last = epoch_loss
            torch.save(model.state_dict(), args.ckpt[0:13] + 'train_best.pth')
        torch.save(model.state_dict(), args.ckpt[0:13] + 'last.pth')
        # Validate the model
        valid_dataset = MSDataset(
            args.val_data_folder, transform=x_transforms, target_transform=y_transforms)
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=True)
        if (epoch + 2) % val_interval == 0:
            loss_val = 0.
            model.eval()   # 关闭dropout模块
            with torch.no_grad():
                step_val = 0
                for x, y in valid_loader:
                    step_val += 1
                    inputs = x.to(args.device)
                    labels = y.to(args.device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_val += loss.item()

                valid_curve.append(loss)
                print("\r epoch %d valid_loss:%0.6f" %
                      (epoch, loss_val / step_val), end='')
            print('\r\n')
            if loss_val <= epoch_valid_loss_last:
                epoch_valid_loss_last = loss_val
                torch.save(model.state_dict(),
                           args.ckpt[0:13] + 'valid_best.pth')
        epoch_best = epoch_loss + loss_val
        if epoch_best <= epoch_best_last:
            epoch_best_last = epoch_best
            torch.save(model.state_dict(), args.ckpt[0:13] + 'best.pth')

    train_x = range(len(train_curve))
    train_y = train_curve

    train_iters = len(dataload)
    valid_x = np.arange(1, len(
        valid_curve) + 1) * train_iters * val_interval  # 由于valid中记录的是EpochLoss，需要对记录点进行转换到iterations
    valid_y = valid_curve

    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.legend(loc='upper right')
    plt.ylabel('loss value')
    plt.xlabel('Iteration')
    plt.savefig(args.ckpt[0:13] + 'loss_value.jpg')

    return model


# 训练模型
def train(args):
    # model = TransUnet(1, 10).to(args.device)
    # model = Unet(1, 10).to(args.device)
    # model = Res2Unet(1, 10).to(args.device)
    # model = Res1Unet(1, 10).to(args.device)
    model = TransRes1Unet(1, 10).to(args.device)
    batch_size = args.batch_size
    # criterion = nn.CrossEntropyLoss()   # 交叉熵
    # criterion = my_loss.MultiClassDiceLoss()   # 多分类dice loss
    # criterion = my_loss.MultiFocalLoss()
    # criterion = my_loss.WeightedDiceCeLoss()
    criterion = my_loss.DiceCeLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ms_dataset = MSDataset(
        args.train_data_folder, transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(
        ms_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(args, model, criterion, optimizer, dataloaders)


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str,
                       help="train, test or dice", default="train")
    parse.add_argument("--train_data_folder", type=str,
                       help="train data", default="./dataset/train/")
    parse.add_argument("--val_data_folder", type=str,
                       help="val data", default="./dataset/val/")
    parse.add_argument("--png_to_predict_folder", type=str,
                       help="png_to_predict", default="./dataset/png_to_predict/")
    parse.add_argument("--pre_output", type=str, help="png predicted output folder",
                       default=r"E:\seg_medical_net\predict\test13")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file",
                       default="./model/exp0/train_best.pth")
    parse.add_argument("--device", type=str, default='cuda')
    parse.add_argument("--batch_size", type=int, default=32)
    parse.add_argument("--epochs", type=int, default=100)
    args = parse.parse_args()

    if args.action == "train":
        train(args)
