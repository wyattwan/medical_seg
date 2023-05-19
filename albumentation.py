import os
import random
import argparse
import cv2
import albumentations as A
from PIL import Image
from matplotlib import pyplot as plt


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.show()

    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        plt.show()


def my_train_aug(mask_folder, origin_folder, aug_nums):
    nums = len(os.listdir(mask_folder))
    random_numbers = [random.randint(0, nums) for _ in range(aug_nums)]

    for i in range(aug_nums):
        image = cv2.imread(origin_folder + "/%d.png" % random_numbers[i])
        mask = cv2.imread(mask_folder + "/%d.png" % random_numbers[i])

        augmentation = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=0.5),  # 镜像
                A.Rotate(limit=20, p=0.4,
                         border_mode=cv2.BORDER_CONSTANT, value=0),  # 旋转
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.05, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2)
            ], p=0.8),
            A.RandomBrightnessContrast(p=0.6),  # 随机明亮对比度
        ], additional_targets={"mask": "mask"})

        augmented_images = augmentation(image=image, mask=mask)
        augmented_image_origin = augmented_images['image']
        augmented_image_mask = augmented_images['mask']

        save_name = os.path.join(mask_folder, "%d.png" % (i + nums))
        Image.fromarray(augmented_image_mask).save(save_name)

        save_name = os.path.join(origin_folder, "%d.png" % (i + nums))
        Image.fromarray(augmented_image_origin).save(save_name)
        i = i + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path of the data to use')
    parser.add_argument('--train_png_mask', type=str,
                        help='converted mask png', default="./dataset/train/png_mask")
    parser.add_argument('--train_png_origin', type=str,
                        help='converted origin png', default="./dataset/train/png_origin")
    parser.add_argument('--samples', type=int, help='nums of aug', default=200)
    args = parser.parse_args()  # 获取所有参数

    my_train_aug('E:\desktop\kkk\mask', 'E:\desktop\kkk\origin', 1)
    print("\n --------successful-------")
