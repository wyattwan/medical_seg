import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from numpy.ma import masked_array
import argparse
import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from common_tools import transform_invert
from common_tools import colormap
from dataset import MSDataset
from torchvision.transforms import transforms
from dataset import onehot2mask
from dataset import mask2onehot


def single_dice_bins_draw(dict):
    descriptions = []
    dice_mean_list_all = []
    save_name = ''
    labels = ['MS_empty', 'MS_mucous', 't2',
              't3', 't4', 't5', 't6', 't7', 't8']   # 9个类别
    for model_key, description in dict.items():
        dice_values_save = './model/' + model_key[8:14] + '/dice_single/'
        dice_means_list = []
        for i in range(1, 10):
            with open(dice_values_save + str(i) + '.txt', "r") as csvfile:
                print(' read' + dice_values_save + str(i) + '.txt')
                reader = csv.reader(csvfile)
                dice_read = next(reader)
                dice_read = [float(x) for x in dice_read]
                dice_means_list.append(sum(dice_read) / len(dice_read))
        descriptions.append(description)
        save_name = save_name + description + '_'
        dice_mean_list_all.append(dice_means_list)

    bar_width = 0
    # 创建图形对象和子图对象
    fig, ax = plt.subplots()
    for i in range(len(dice_mean_list_all)):
        values = dice_mean_list_all[i]
        print(values)
        if i == 0:
            ax.bar(labels, values, width=0.1, label=descriptions[i])
        else:
            # 绘制第二个柱状图
            ax.bar([x + bar_width for x in range(len(values))],
                   values, width=0.1, label=descriptions[i])
        bar_width += 0.1
    # 添加标签和标题
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('precision')

    # 添加图例
    ax.legend(loc='lower right')
    plt.xticks(rotation=15)
    plt.savefig('./predict/dice_bins_compare/' + save_name + '_precision.png')
    plt.show()


def dice_bins_draw(dict):
    dice_lists_all = []
    dice_means_list = []
    descriptions = []
    colors = ['red', 'blue', 'green', 'orange',
              'purple', 'pink', 'brown', 'gray', 'black']
    save_name = ''
    for model_key, description in dict.items():
        dice_values_save = './model/' + model_key[8:14]
        with open(dice_values_save + description + '.txt', "r") as csvfile:
            print(' read' + dice_values_save + description + '.txt')
            reader = csv.reader(csvfile)
            dice_read = next(reader)
            dice_read = [float(x) for x in dice_read]
            dice_means_list.append(sum(dice_read) / len(dice_read))
            dice_lists_all.append(dice_read)
        descriptions.append(description)
        save_name = save_name + description + '_'
    plt.hist(dice_lists_all,
             bins=10,
             rwidth=1,
             label=descriptions,
             color=colors[0:len(descriptions)])
    plt.legend(loc='upper left')
    plt.xlabel('dice coefficient bins')  # 绘制x轴
    plt.ylabel('Num of test scans')  # 绘制y轴
    plt.grid(linestyle='--', linewidth=0.3)
    plt.savefig('./predict/dice_bins_compare/' + save_name + '_dice.png')
    plt.show()


def mean_dice_draw(dict):
    dice_means_list = []
    descriptions = []
    # colors = ['red', 'blue', 'green', 'orange',
    #           'purple', 'pink', 'brown', 'gray', 'black']
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # 为每个柱子指定颜色
    # colors = ['red', 'blue']
    # labels = ['focal', 'LogDiceCe']
    labels = ['unet', 'res1', 'res2', 'transres2', 'transres1']
    save_name = ''
    for model_key, description in dict.items():
        dice_values_save = './model/' + model_key[8:14]
        with open(dice_values_save + description + '_dicemean.txt', "r") as csvfile:
            print(' read' + dice_values_save + description + '.txt')
            reader = csv.reader(csvfile)
            dice_read = next(reader)
            dice_read = [float(x) for x in dice_read]
            dice_means_list.append(sum(dice_read) / len(dice_read))
        descriptions.append(description)
        save_name = save_name + description + '_'
    # print(dice_means_list)

    # 创建柱状图
    fig, ax = plt.subplots()
    bar_width = 0.5  # 柱子的宽度
    x_pos = np.arange(len(dice_means_list))

    # 在x_pos位置添加数据，设置宽度为bar_width，添加标签
    bars = ax.bar(x_pos, dice_means_list, width=bar_width, tick_label=labels, color=colors)

    # 添加标签和标题
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')
    ax.set_title('mean dice')   

    # 添加图例
    ax.legend(loc='lower right')
    plt.xticks(rotation=15)
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            '{:.4f}'.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords='offset points',
            ha='center', va='bottom'
        )

    # 保存图像到文件
    plt.savefig('./predict/dice_bins_compare/' + save_name + '_meandice.png')
    plt.show()


if __name__ == '__main__':
    # 参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--png_to_test_folder", type=str,
                       help="png_to_predict", default="./dataset/test/")
    parse.add_argument("--device", type=str, default='cuda')
    args = parse.parse_args()

    model_Res1Net_dict = {"./model/exp8/valid_best.pth": "res1_focalLoss"}

    # model_dict = {"./model/exp12/valid_best.pth": "res2_ratioDiceCeLoss",
    #               "./model/exp13/valid_best.pth": "res2_logDiceCeLoss",
    #               "./model/exp14/valid_best.pth": "new_data_with_res2_logDiceCeLoss",
    #               "./model/exp15/valid_best.pth": "new_data_with_res2_RatioDiceCeLoss",
    #               "./model/exp16/valid_best.pth": "new_data_with_res2_logDiceCeLoss_without_dropout",
    #               "./model/exp17/valid_best.pth": "trans_ratioDiceCeLoss"}

    # "./model/exp19/best.pth": "res1_focalLoss"
    model_dict = {"./model/exp18/best.pth": "unet_logDiceCeLoss",
                  "./model/exp20/best.pth": "res1_logDiceCeLoss",
                  "./model/exp21/valid_best.pth": "res2_logDiceCeLoss",
                  "./model/exp22/valid_best.pth": "transres2_LogDiceCeLoss",
                  "./model/exp23/best.pth": "transres1_LogDiceCeLoss"}
    # model_dict = {
    #     "./model/exp19/best.pth": "res1_focalLoss",
    #     "./model/exp20/best.pth": "res1_logDiceCeLoss"
    # }
    single_dice_bins_draw(model_dict)
    dice_bins_draw(model_dict)
    mean_dice_draw(model_dict)

    print(" analysis successfully")
