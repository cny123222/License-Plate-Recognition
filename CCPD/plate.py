import yolov5
import cv2
import os
from tqdm import tqdm
import csv
import torch
from concurrent.futures import ThreadPoolExecutor
import math

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 定义字符映射规则
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 确定设备
if torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_ccpd_filename(filename):
    """
    从 CCPD 文件名中解析车牌号
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('-')
    char_indices = list(map(int, parts[4].split('_')))
    plate_number = provinces[char_indices[0]] + \
                   alphabets[char_indices[1]] + \
                   ''.join([ads[idx] for idx in char_indices[2:]])
    return plate_number


def traverse_images(split_file, base_dir):
    """
    根据 train.txt 或 val.txt 遍历图片路径
    """
    with open(split_file, 'r') as f:
        image_paths = [os.path.join(base_dir, line.strip()) for line in f]
    return image_paths


def process_single_image(image_path, model, output_dir, writer):
    """
    单独处理一张图片，裁剪车牌并保存，同时生成标签
    """

    # 保存裁剪结果
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cropped_plate)

    # 获取车牌号
    plate_number = parse_ccpd_filename(filename)
    if plate_number is None:  # 如果解析失败，跳过
        return

    # 写入标签
    writer.writerow([output_path, plate_number])


def crop_and_save_plates(image_paths, output_dir, output_csv):
    """
    使用 YOLO 模型逐张裁剪车牌并保存，同时生成标签
    """
    # 加载 YOLO 模型
    model = yolov5.load('keremberke/yolov5n-license-plate')
    # model = model.to(device)
    model.conf = 0.25
    model.iou = 0.45
    os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])  # 写入表头

        for image_path in tqdm(image_paths, desc="Processing images"):
            process_single_image(image_path, model, output_dir, writer)


def crop_and_save_plates_parallel(image_paths, output_dir, output_csv, num_workers=4):
    """
    使用多线程或多进程加速逐张裁剪车牌并保存，同时生成标签
    """
    # 加载 YOLO 模型（注意：模型需要在每个线程中加载一次）
    model = yolov5.load('keremberke/yolov5n-license-plate')
    model.conf = 0.25
    model.iou = 0.45
    os.makedirs(output_dir, exist_ok=True)

    def process_image(image_path):
        """处理单张图片"""
        try:
            process_single_image(image_path, model, output_dir, writer)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    # 分配任务到多线程
    chunk_size = math.ceil(len(image_paths) / num_workers)
    chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])  # 写入表头

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Processing images"))


# 配置路径
ccpd_base_dir = "CCPD2019"
train_split = os.path.join(ccpd_base_dir, "splits/train.txt")
val_split = os.path.join(ccpd_base_dir, "splits/val.txt")
train_output_dir = "CCPD/cropped/train"
val_output_dir = "CCPD/cropped/val"
train_output_csv = "CCPD/train.csv"
val_output_csv = "CCPD/val.csv"

# 执行裁剪与生成标签
# train_image_paths = traverse_images(train_split, ccpd_base_dir)
# crop_and_save_plates(train_image_paths, train_output_dir, train_output_csv)

num_workers = 4  # 根据 CPU 核心数调整
train_image_paths = traverse_images(train_split, ccpd_base_dir)
crop_and_save_plates_parallel(train_image_paths, train_output_dir, train_output_csv, num_workers=num_workers)

val_image_paths = traverse_images(val_split, ccpd_base_dir)
crop_and_save_plates_parallel(val_image_paths, val_output_dir, val_output_csv, num_workers=num_workers)