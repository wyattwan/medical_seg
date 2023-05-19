"""将dicom数据转换为png数据"""
import os
import pydicom
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as f
from matplotlib import pylab
from skimage import img_as_float
from PIL import Image
import shutil
import albumentation

dicom_batch_array = []
dicom_batch_array0 = []


def dicom_merge(args):   # 把散的mask合成一张多分类的灰度图
    # 读病人编号, patient0  patient1 ....文件夹列表
    patient_names = os.listdir(args.patient_folder)
    dicom_batch_array.clear()
    dicom_batch_array0.clear()
    print(" -------------------------")
    print(" " + str(len(patient_names)) +
          " patient found in " + args.patient_folder + '\r\n')
    print(" mask merge")
    for i in range(len(patient_names)):
        if patient_names[i] != 'patient_empty':
            single_patient_name = os.path.join(
                args.patient_folder, patient_names[i])   # 单个病人文件夹, /patient
            # 列出器官名, MS_empty  MS-mucous ...文件夹列表
            organ_names = os.listdir(single_patient_name)
            dicom_batch_array.clear()
            dicom_batch_array0.clear()
            length = 0  # 每个器官dicom文件数
            index_class = 1
            width = 0
            height = 0
            convert_judge = os.path.join(
                single_patient_name, 'mask_merge_png')   # 融合后文件夹
            if len(os.listdir(convert_judge)) == 0:   # 已经转换过的不用再转换
                for j in range(len(organ_names)):
                    lock = 0  # 记录读图是否完成
                    singe_organ_name = os.path.join(
                        single_patient_name, organ_names[j])    # 具体到每个器官文件夹
                    # 不对原图像dicom进行处理
                    if organ_names[j] != 'dicom_origin' and organ_names[j] != 'mask_merge_png' and organ_names[j] != 'origin_png':
                        dicom_files = os.listdir(
                            singe_organ_name)   # 具体单个器官文件夹下的dicoms
                        print(" " + str(len(dicom_files)) +
                              " dicoms found in " + singe_organ_name)
                        length = length if length > len(
                            dicom_files) else len(dicom_files)
                        for k in range(length):   # 遍历dicoms
                            if len(dicom_files) != 0:
                                single_dicom_path = os.path.join(
                                    singe_organ_name, dicom_files[k])
                                dicom = pydicom.read_file(
                                    single_dicom_path)   # 读取每一个dicom
                                dicom_image_array = dicom.pixel_array.astype(
                                    float)   # 得到图像矩阵
                                width = np.size(dicom_image_array, 0)
                                height = np.size(dicom_image_array, 1)
                            elif len(dicom_files) == 0:
                                dicom_image_array = np.zeros((width, height))
                            if dicom_image_array.max() != 0:
                                scaled_image = (np.maximum(
                                    dicom_image_array, 0) / dicom_image_array.max()) * 255.0  # 归一化
                                if organ_names[j] == 'MS_empty' and index_class == 1:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 1
                                    lock += 1
                                elif organ_names[j] == 'MS_mucous' and index_class == 2:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 2
                                    lock += 1
                                elif organ_names[j] == 't2' and index_class == 3:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 3
                                    lock += 1
                                elif organ_names[j] == 't3' and index_class == 4:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 4
                                    lock += 1
                                elif organ_names[j] == 't4' and index_class == 5:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 5
                                    lock += 1
                                elif organ_names[j] == 't5' and index_class == 6:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 6
                                    lock += 1
                                elif organ_names[j] == 't6' and index_class == 7:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 7
                                    lock += 1
                                elif organ_names[j] == 't7' and index_class == 8:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 8
                                    lock += 1
                                elif organ_names[j] == 't_wisdom' and index_class == 9:
                                    scaled_image[scaled_image ==
                                                 scaled_image.max()] = 9
                                    lock += 1
                            elif dicom_image_array.max() == 0:
                                scaled_image = np.uint8(dicom_image_array)
                                lock += 1
                            scaled_image = np.uint8(scaled_image)   # 数值转为int
                            if lock == length:
                                index_class += 1
                            dicom_batch_array.append(
                                scaled_image)   # 用列表存放numpy数组

                for m in range(length):
                    nums = int(len(dicom_batch_array) / length)   # 全部class
                    dicom_batch_array0.clear()
                    dicom_batch_array0.append(np.zeros(
                        (np.size(dicom_batch_array[0], 0), np.size(dicom_batch_array[0], 1))))  # 创建空索引
                    for n in range(nums):
                        dicom_batch_array0.append(
                            dicom_batch_array[m + n * length])
                    dicom_batch_array[m] = np.argmax(
                        dicom_batch_array0, axis=0).astype(np.uint8)
                    for a in range(nums):
                        dicom_batch_array[m][dicom_batch_array[m]
                                             == nums - a] = nums - a
                    image0 = Image.fromarray(dicom_batch_array[m])
                    resize = transforms.Resize([args.width, args.width], interpolation=f._interpolation_modes_from_int(
                        0))  # 使用transform变形缩放作为输入，保证推理时和现在缩放方式一致问题应该不会太大
                    image0 = resize(image0)
                    save_path = os.path.join(
                        single_patient_name, 'mask_merge_png')
                    # 生产具体保存名  xxx/xxx/xxx.png
                    save_name = os.path.join(save_path, (str(m) + '.png'))
                    image0.save(save_name)
            else:
                print(' ' + str(patient_names[i]) + ' mask has been converted')


def dicom2png(args):  # 将dicom原图像转换为png格式
    # 读病人编号, patient0  patient1 ....文件夹列表
    patient_names = os.listdir(args.patient_folder)
    print(" -------------------------")
    print(" " + str(len(patient_names)) +
          " patient found in " + args.patient_folder + '\r\n')
    print(" origin convert")
    for i in range(len(patient_names)):
        if patient_names[i] != 'patient_empty':
            single_patient_name = os.path.join(
                args.patient_folder, patient_names[i])  # 单个病人文件夹, /patient
            # 列出器官名, MS_empty  MS-mucous ...文件夹列表
            organ_names = os.listdir(single_patient_name)
            convert_judge = os.path.join(
                single_patient_name, 'origin_png')  # 融合后文件夹
            if len(os.listdir(convert_judge)) == 0:  # 已经转换过的不用再转换
                for j in range(len(organ_names)):
                    singe_organ_name = os.path.join(
                        single_patient_name, organ_names[j])  # 具体到每个器官文件夹
                    if organ_names[j] == 'dicom_origin':  # 对原图像dicom进行处理
                        dicom_files = os.listdir(
                            singe_organ_name)  # 具体单个器官文件夹下的dicoms
                        print(" " + str(len(dicom_files)) +
                              " dicoms found in " + singe_organ_name)
                        for k in range(len(dicom_files)):  # 遍历dicoms
                            singe_dicom_file_name = os.path.join(
                                singe_organ_name, dicom_files[k])
                            dicom = pydicom.read_file(
                                singe_dicom_file_name, force=True)  # 读取当前遍历的单个dicom文件
                            dicom_image_array = dicom.pixel_array.astype(float)
                            if dicom_image_array.max() != 0:
                                scaled_image = (np.maximum(
                                    dicom_image_array, 0) / dicom_image_array.max()) * 255.0  # 归一化
                            elif dicom_image_array.max() == 0:
                                scaled_image = np.uint8(dicom_image_array)
                            scaled_image = np.uint8(scaled_image)
                            image0 = Image.fromarray(scaled_image)

                            # 使用transform变形缩放作为输入，保证推理时和现在缩放方式一致问题应该不会太大
                            resize = transforms.Resize(
                                [args.width, args.width])
                            image0 = resize(image0)
                            save_path = os.path.join(
                                single_patient_name, 'origin_png')
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(k) + '.png'))
                            image0.save(save_name)
            else:
                print(' ' + str(patient_names[i]) +
                      ' origin has been converted')


def png_prepare(args):
    # 读病人编号, patient0  patient1 ....文件夹列表
    patient_names = os.listdir(args.patient_folder)
    train_list = [0, 1, 4, 6, 8, 9, 10, 11, 13, 14, 16,
                  17, 18, 19, 20, 21, 22, 23, 25, 26, 28, 29, 30, 31, 34, 36]
    valid_list = [3, 5, 7, 24, 27, 33]
    test_list = [2, 12, 15, 32, 35]
    m, n = 0, 0
    a, b = 0, 0
    c, d = 0, 0
    print(" -------------------------")
    print(" " + str(len(patient_names)) +
          " patient found in " + args.patient_folder + '\r\n')
    print(" begin to divide datas ")
    for i in range(len(patient_names)):
        if patient_names[i] != 'patient_empty':
            divided_number = int(patient_names[i][7:9])
            if divided_number in train_list:
                single_patient_name = os.path.join(
                    args.patient_folder, patient_names[i])  # 单个病人文件夹路径, /patient
                # 列出器官名, MS_empty  MS-mucous ...文件夹列表
                organ_names = os.listdir(single_patient_name)
                for j in range(len(organ_names)):
                    singe_organ_name = os.path.join(
                        single_patient_name, organ_names[j])  # 具体到每个器官文件夹path
                    if organ_names[j] == 'mask_merge_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.train_png_mask
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(m) + '.png'))
                            m += 1
                            img_mask.save(save_name)
                    elif organ_names[j] == 'origin_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.train_png_origin
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(n) + '.png'))
                            n += 1
                            img_mask.save(save_name)
            elif divided_number in valid_list:
                single_patient_name = os.path.join(
                    args.patient_folder, patient_names[i])  # 单个病人文件夹路径, /patient
                # 列出器官名, MS_empty  MS-mucous ...文件夹列表
                organ_names = os.listdir(single_patient_name)
                for j in range(len(organ_names)):
                    singe_organ_name = os.path.join(
                        single_patient_name, organ_names[j])  # 具体到每个器官文件夹path
                    if organ_names[j] == 'mask_merge_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.val_png_mask
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(a) + '.png'))
                            a += 1
                            img_mask.save(save_name)
                    elif organ_names[j] == 'origin_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.val_png_origin
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(b) + '.png'))
                            b += 1
                            img_mask.save(save_name)
            elif divided_number in test_list:
                single_patient_name = os.path.join(
                    args.patient_folder, patient_names[i])  # 单个病人文件夹路径, /patient
                # 列出器官名, MS_empty  MS-mucous ...文件夹列表
                organ_names = os.listdir(single_patient_name)
                for j in range(len(organ_names)):
                    singe_organ_name = os.path.join(
                        single_patient_name, organ_names[j])  # 具体到每个器官文件夹path
                    if organ_names[j] == 'mask_merge_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.test_png_mask
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(c) + '.png'))
                            c += 1
                            img_mask.save(save_name)
                    elif organ_names[j] == 'origin_png':  # 如果是融合的后的png图片
                        # 具体融合png_mask下的pngs
                        png_files = os.listdir(singe_organ_name)
                        png_files.sort(key=lambda x: int(x.split('.')[0]))
                        print(" " + str(len(png_files)) +
                              " pngs found in " + singe_organ_name)
                        for k in range(len(png_files)):  # 遍历pngs
                            singe_png_file_name = os.path.join(
                                singe_organ_name, png_files[k])
                            img_mask = Image.open(singe_png_file_name)
                            save_path = args.test_png_origin
                            # 生产具体保存名  xxx/xxx/xxx.png
                            save_name = os.path.join(
                                save_path, (str(d) + '.png'))
                            d += 1
                            img_mask.save(save_name)


def dicom_patient_tags_get(singe_dicom_file_name):  # 常用dicom 病人tags获取
    dicom = pydicom.read_file(singe_dicom_file_name)
    print(dicom.PatientName, dicom.PatientSex,
          dicom.PatientID)  # 打印对应dicom的tag数值
    # print(dicom.dir())  # dicom文件的具体tags


def dicom_pixel_tags_get(singe_dicom_file_name):
    dicom = pydicom.read_file(singe_dicom_file_name)
    print(dicom.Columns, dicom.Rows)  # 打印对应dicom的tag数值
    print(dicom.PixelSpacing)  # 像素点间距，用于计算距离
    print(dicom.BitsAllocated)  # 像素点存储位数,一般为0-255，即256级灰度
    print(len(dicom.PixelData))  # 长度为columns * rows 的一维二进制列表


# 用matplotlib显示dicom的single切片
def dicom_image_show_in_mat(singe_dicom_file_path):
    dicom = pydicom.read_file(singe_dicom_file_path)
    image = dicom.pixel_array
    plt.imshow(image, cmap=pylab.cm.bone)
    plt.show()


def remove_dir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='path of the data to use')
    parser.add_argument('--train_png_mask', type=str,
                        help='converted mask png', default="./dataset/train/png_mask")
    parser.add_argument('--train_png_origin', type=str,
                        help='converted origin png', default="./dataset/train/png_origin")
    parser.add_argument('--val_png_mask', type=str,
                        help='converted mask png', default="./dataset/val/png_mask")
    parser.add_argument('--val_png_origin', type=str,
                        help='converted origin png', default="./dataset/val/png_origin")
    parser.add_argument('--test_png_mask', type=str,
                        help='converted mask png', default="./dataset/test/png_mask")
    parser.add_argument('--test_png_origin', type=str,
                        help='converted origin png', default="./dataset/test/png_origin")
    parser.add_argument('--patient_folder', type=str,
                        default="./dataset/mimics_patient/")
    parser.add_argument('--width', type=int, default=224)
    args = parser.parse_args()  # 获取所有参数

    t2_patients = [0, 8, 9, 11, 13, 17, 19, 26]
    t8_patients = [4, 6, 8, 9, 10, 14, 16, 17, 19, 21, 25, 28, 29, 30]
    for i in range(len(t2_patients)):
        patient_mask_path = './dataset/mimics_patient/patient' + \
            str(t2_patients[i]) + '/mask_merge_png'
        patient_origin_path = './dataset/mimics_patient/patient' + \
            str(t2_patients[i]) + '/origin_png'
        print(' empty the ' + './dataset/mimics_patient/patient' +
              str(t2_patients[i]) + ' now')
        remove_dir(patient_mask_path)
        remove_dir(patient_origin_path)
    for j in range(len(t8_patients)):
        patient_mask_path = './dataset/mimics_patient/patient' + \
            str(t8_patients[j]) + '/mask_merge_png'
        patient_origin_path = './dataset/mimics_patient/patient' + \
            str(t8_patients[j]) + '/origin_png'
        print(' empty the ' + './dataset/mimics_patient/patient' +
              str(t8_patients[j]) + ' now')
        remove_dir(patient_mask_path)
        remove_dir(patient_origin_path)

    dicom2png(args)  # 将原图像转为png图像保存在对应文件夹下

    dicom_merge(args)

    # 对二号牙和八号牙进行单独数据增强
    for i in range(len(t2_patients)):
        patient_mask_path = './dataset/mimics_patient/patient' + \
            str(t2_patients[i]) + '/mask_merge_png'
        patient_origin_path = './dataset/mimics_patient/patient' + \
            str(t2_patients[i]) + '/origin_png'
        print(' augment the ' + './dataset/mimics_patient/patient' +
              str(t2_patients[i]) + ' now')
        albumentation.my_train_aug(patient_mask_path, patient_origin_path, 200)
    for j in range(len(t8_patients)):
        patient_mask_path = './dataset/mimics_patient/patient' + \
            str(t8_patients[j]) + '/mask_merge_png'
        patient_origin_path = './dataset/mimics_patient/patient' + \
            str(t8_patients[j]) + '/origin_png'
        print(' augment the ' + './dataset/mimics_patient/patient' +
              str(t8_patients[j]) + ' now')
        albumentation.my_train_aug(patient_mask_path, patient_origin_path, 200)

    # 准备新数据集之前清空之前的数据集
    remove_dir(args.train_png_mask)
    remove_dir(args.train_png_origin)
    remove_dir(args.val_png_mask)
    remove_dir(args.val_png_origin)
    remove_dir(args.test_png_mask)
    remove_dir(args.test_png_origin)
    png_prepare(args)

    albumentation.my_train_aug(
        args.train_png_mask, args.train_png_origin, 5000)

    print("\n -------dicom-init-successful-------")
