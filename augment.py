import os
import Augmentor
import argparse
from PIL import Image
from PIL import ImageFilter
import shutil


def my_train_aug(args):
    m, n = 0, 0
    nums = len(os.listdir(args.train_png_origin))
    p = Augmentor.Pipeline(args.train_png_origin)
    p.ground_truth(args.train_png_mask)

    # # 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
    # p.rotate(probability=0.8, max_left_rotation=5, max_right_rotation=5)

    # 图像左右互换： 按照概率0.9执行
    p.flip_left_right(probability=0.9)

    # 最终扩充的数据样本数, origin和mask各4000
    p.sample(args.samples)
    print(" create-augmented-pngs-successful")

    print(" start to copy and paste")
    output_dir = os.path.join(args.train_png_origin, "output")
    output_names = os.listdir(output_dir)
    for i in range(len(output_names)):
        if output_names[i][0:7] == "_ground":
            used_name = os.path.join(output_dir, output_names[i])
            save_name = os.path.join(
                args.train_png_mask, "%d.png" % (m + nums))
            image = Image.open(used_name)
            image.save(save_name)
            m = m + 1

        if output_names[i][0:3] == "png":
            used_name = os.path.join(output_dir, output_names[i])
            save_name = os.path.join(
                args.train_png_origin, "%d.png" % (n + nums))
            image = Image.open(used_name)
            image.save(save_name)
            n = n + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path of the data to use')
    parser.add_argument('--train_png_mask', type=str,
                        help='converted mask png', default="./dataset/train/png_mask")
    parser.add_argument('--train_png_origin', type=str,
                        help='converted origin png', default="./dataset/train/png_origin")
    parser.add_argument('--samples', type=int,
                        help='nums of aug', default=3000)
    args = parser.parse_args()  # 获取所有参数

    my_train_aug(args)

    output_dir = os.path.join(args.train_png_origin, "output")
    shutil.rmtree(output_dir)
    print("\n --------successful-------")
