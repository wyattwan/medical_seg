import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from numpy.ma import masked_array
import argparse
import cv2
import os
import csv
import my_loss
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common_tools import transform_invert
from common_tools import colormap
from unet import TransUnet
from unet import Res1Unet
from unet import Res2Unet
from unet import Unet
from unet import TransRes1Unet
from dataset import MSDataset
from torchvision.transforms import transforms
from dataset import onehot2mask
from dataset import mask2onehot


x_transforms = transforms.Compose([
    transforms.ToTensor(),
])
y_transforms = None


def get_confusion_matrix(ground_truth, prediction):
    # gt_onehot = torch.nn.functional.one_hot(ground_truth, num_classes=num_classes)
    # pd_onehot = torch.nn.functional.one_hot(prediction, num_classes=num_classes)
    return prediction.t().matmul(ground_truth)


class Colorize:
    def __init__(self, n=10):
        self.cmap = colormap(224)
        self.cmap = torch.from_numpy(self.cmap[:n])  # array->tensor

    def __call__(self, mask0):
        size = mask0.size()  # 这里就是上文的output
        color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (mask0 == label)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


# cal dice when detecting, and save the output in colorful map
def detect(args, dict, model_index):
    if model_index == 0:
        model = Unet(1, 10).to(args.device)
    elif model_index == 1:
        model = Res1Unet(1, 10).to(args.device)
    elif model_index == 2:
        model = Res2Unet(1, 10).to(args.device)
    elif model_index == 3:
        model = TransUnet(1, 10).to(args.device)
    elif model_index == 4:
        model = TransRes1Unet(1, 10).to(args.device)

    for model_key, description in dict.items():
        ms_dataset = MSDataset(
            args.png_to_test_folder, transform=x_transforms, target_transform=y_transforms)
        dataloaders = DataLoader(ms_dataset, batch_size=1)
        nums = ms_dataset.__len__()
        model.load_state_dict(torch.load(model_key, map_location=args.device))
        save_root = './predict/' + model_key[8:13]
        if model_index == 0:
            dice_values_save = './model/' + model_key[8:14]
        elif model_index == 1:
            dice_values_save = './model/' + model_key[8:14]
        elif model_index == 2 or model_index == 3:
            dice_values_save = './model/' + model_key[8:14]
        elif model_index == 3:
            dice_values_save = './model/' + model_key[8:14]
        elif model_index == 4:
            dice_values_save = './model/' + model_key[8:14]
        print('\r\n')
        print(' start to test with' + model_key +
              ' and save the images in' + save_root)
        model.eval()   # 关闭dropout
        dice_scores = [[], [], [], [], [], [], [], [], []]
        dice_mean_list = []
        dice_list = []
        index = 0
        with torch.no_grad():
            # with (1, 224, 224) - (10, 224, 224)
            true_positives_all = 0 
            false_positives_all = 0
            false_negatives_all = 0
            for x, ground in dataloaders:
                colorize = Colorize(10)
                # criterion = my_loss.DiceCeLoss()
                # output shape : (10, 224, 224)
                y = model(x.to(args.device)).to('cpu')
                # loss = criterion(y, ground)

                img_y = torch.squeeze(y).numpy()
                img_y = onehot2mask(img_y)
                img_predict = mask2onehot(img_y, 10)
                img_y = torch.from_numpy(img_y)
                img_y = colorize(img_y)
                img_y = torch.transpose(img_y, 0, 2)
                img_y = torch.transpose(img_y, 0, 1).numpy()

                img_ground = torch.squeeze(ground).numpy()
                img_ground = onehot2mask(img_ground)
                img_ground1 = mask2onehot(img_ground, 10)
                img_ground = torch.from_numpy(img_ground)
                img_ground = colorize(img_ground)
                img_ground = torch.transpose(img_ground, 0, 2)
                img_ground = torch.transpose(img_ground, 0, 1).numpy()

                # Convert the masks to PyTorch tensors
                predicted_mask = torch.tensor(img_predict)
                ground_truth_mask = torch.tensor(img_ground1)

                true_positives = 0
                false_positives = 0
                false_negatives = 0
                for classes in range(1, 10):
                    true_positive = torch.sum(
                        torch.logical_and(predicted_mask[classes] == 1, ground_truth_mask[classes] == 1))
                    false_positive = torch.sum(
                        torch.logical_and(predicted_mask[classes] == 1, ground_truth_mask[classes] == 0))
                    false_negative = torch.sum(
                        torch.logical_and(predicted_mask[classes] == 0, ground_truth_mask[classes] == 1))
                    true_positives += true_positive
                    false_positives += false_positive
                    false_negatives += false_negative
                    if false_positive.item() != 0 or false_negative.item() != 0:
                        precision = (true_positive.item() + 1e-6) / \
                            (true_positive.item() + false_positive.item() + 1e-6)
                        # recall = true_positive.item() / (true_positive.item() + false_negative.item())
                        # Calculate precision, recall, and Dice coefficient for each dimension
                        # dice = 2 * true_positive.item() / (
                        #             2 * true_positive.item() + false_positive.item() + false_negative.item())
                        dice_scores[classes - 1].append(precision)

                true_positives_all += true_positives.item()
                false_positives_all += false_positives.item()
                false_negatives_all += false_negatives.item()
    
                if false_positives.item() != 0 or false_negatives.item() != 0:
                    dice = 2 * true_positives.item() / (2 * true_positives.item() +
                                                        false_positives.item() + false_negatives.item())
                    dice_list.append(dice)

                x = torch.squeeze(x)
                x = x.unsqueeze(0)
                img_x = transform_invert(x, x_transforms)   # 从torch数据退回图片数据

                src_path = os.path.join(save_root, "predict_%d_s.png" % index)
                save_path = os.path.join(save_root, "predict_%d_o.png" % index)
                ground_path = os.path.join(
                    save_root, "predict_%d_g.png" % index)

                img_x.save(src_path)
                cv2.imwrite(save_path, img_y)
                cv2.imwrite(ground_path, img_ground)
                index = index + 1
                print("\r image round %d / %d" % (index, nums), end='')

            dice_mean = 2 * true_positives_all / (2 * true_positives_all +
                                    false_positives_all + false_negatives_all)
            dice_mean_list.append(dice_mean)

            for i in range(0, 9):
                open(dice_values_save + 'dice_single/' + str(i + 1) + '.txt', 'w')
                print('\r\n')
                print(' save dice values in ' + dice_values_save +
                      'dice_single/' + str(i + 1) + '.txt')
                with open(dice_values_save + 'dice_single/' + str(i + 1) + '.txt', "w") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(dice_scores[i])
            open(dice_values_save + description + '.txt', 'w')
            print('\r\n')
            print(' save dice values in ' +
                  dice_values_save + description + '.txt')
            with open(dice_values_save + description + '.txt', "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(dice_list)
            print('\r\n')
            print(' save dice mean values in ' +
                  dice_values_save + description + '_dicemean.txt')
            with open(dice_values_save + description + '_dicemean.txt', "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(dice_mean_list)


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--png_to_test_folder", type=str,
                       help="png_to_predict", default="./dataset/test/")
    parse.add_argument("--device", type=str, default='cuda')
    args = parse.parse_args()

    # model_Res1Net_dict = {"./model/exp8/valid_best.pth": "res1_focalLoss"}
    # model_Res2Net_dict = {"./model/exp12/valid_best.pth": "res2_ratioDiceCeLoss",
    #                       "./model/exp13/valid_best.pth": "res2_logDiceCeLoss",
    #                       "./model/exp14/valid_best.pth": "new_data_with_res2_logDiceCeLoss",
    #                       "./model/exp15/valid_best.pth": "new_data_with_res2_RatioDiceCeLoss",
    #                       "./model/exp16/valid_best.pth": "new_data_with_res2_logDiceCeLoss_without_dropout"}

    model_Res1Net_FocalLoss = {"./model/exp19/best.pth": "res1_focalLoss"}
    model_Unet_LogDiceCe = {"./model/exp18/best.pth": "unet_logDiceCeLoss"}
    model_Res1Net_LogDiceCeLoss = {"./model/exp20/best.pth": "res1_logDiceCeLoss"}
    model_Res2Net_LogDiceCeLoss = {"./model/exp21/valid_best.pth": "res2_logDiceCeLoss"}
    model_TransRes2Net_dict = {"./model/exp22/valid_best.pth": "transres2_LogDiceCeLoss"}
    model_TransRes1Net_dict = {"./model/exp23/valid_best.pth": "transres1_LogDiceCeLoss"}

    # 0-Unet 1-Res1Net 2-Res2Net 3-TransRes2Net 4-TransRes1Net
    detect(args, model_Unet_LogDiceCe, 0)
    detect(args, model_Res1Net_FocalLoss, 1)
    detect(args, model_Res1Net_LogDiceCeLoss, 1)
    detect(args, model_Res2Net_LogDiceCeLoss, 2)
    detect(args, model_TransRes2Net_dict, 3)
    detect(args, model_TransRes1Net_dict, 4)

    print(" detect successfully")
