import csv
import os

def parse_ccpd_filename(filename):
    """
    解析 CCPD 数据集的文件名，提取车牌号码及其他关键信息
    :param filename: CCPD 图片文件名
    :return: 车牌号、置信度、矩形框顶点坐标、四个顶点坐标
    """
    # 去掉扩展名
    name = os.path.splitext(filename)[0]
    # 按照 '-' 分割字段
    parts = name.split('-')
    
    # 提取车牌号
    plate_number = parts[0]
    
    # 提取置信度
    confidence = parts[1]
    
    # 提取矩形框的左上和右下顶点坐标
    box_coords = parts[2].split('_')
    top_left = list(map(int, box_coords[0].split('&')))
    bottom_right = list(map(int, box_coords[1].split('&')))
    
    # 提取四个顶点的具体坐标
    corner_coords = list(map(int, parts[3].split('_')))
    corners = [
        (corner_coords[0], corner_coords[1]),  # 左上
        (corner_coords[2], corner_coords[3]),  # 左下
        (corner_coords[4], corner_coords[5]),  # 右下
        (corner_coords[6], corner_coords[7])   # 右上
    ]
    
    return plate_number, confidence, top_left, bottom_right, corners


def generate_labels(cropped_dir, original_dir, output_csv):
    """
    为裁剪后的车牌图像生成标签
    :param cropped_dir: 裁剪后的车牌图像目录
    :param original_dir: 原始 CCPD 数据集目录
    :param output_csv: 输出的标签 CSV 文件路径
    """
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image_path', 'label'])

        for file in os.listdir(cropped_dir):
            if file.endswith('.jpg'):
                # 从原始图像文件名提取车牌号码
                original_filename = file
                plate_number, _, _, _, _ = parse_ccpd_filename(original_filename)

                # 写入裁剪图像路径和车牌号
                image_path = os.path.join(cropped_dir, file)
                writer.writerow([image_path, plate_number])

# 执行生成标签
cropped_dir = "/path/to/cropped_plates"  # 裁剪后的车牌图像目录
original_dir = "/path/to/CCPD/train"  # 原始 CCPD 数据集目录
output_csv = "labels.csv"  # 输出标签文件路径
generate_labels(cropped_dir, original_dir, output_csv)