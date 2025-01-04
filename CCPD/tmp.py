import csv
import re
import os

# 定义合法字符集
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
charset = set(provinces + ads)

# 定义合法标签的正则表达式：省份 + 6个字符
valid_label_pattern = re.compile(f"^[{''.join(provinces)}][{''.join(ads)}]{{5,6}}$")

# 定义合法路径的检查函数
def is_valid_image_path(image_path):
    # 检查路径是否存在
    if not os.path.exists(image_path):
        return False
    # 检查文件扩展名是否合法
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions


def filter_invalid_rows(input_csv, output_csv):
    """
    删除含有非法字符、路径或异常行，并保存合法行到新的文件中
    """
    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader((line.replace('\0', '') for line in infile))  # 移除 NUL 字符
        writer = csv.writer(outfile)

        # 写入表头
        header = next(reader, None)
        if header is None or len(header) != 2:
            print("文件表头异常，确保文件为两列格式！")
            return
        writer.writerow(header)

        # 检查每一行的标签是否合法
        for line_number, row in enumerate(reader, start=2):  # 行号从 2 开始（因为跳过表头）
            try:
                if len(row) != 2:
                    print(f"跳过格式错误的行: 第 {line_number} 行, 内容: {row}")
                    continue

                image_path = row[0].strip()
                label = row[1].strip()

                # 检查 image_path 和 label 的合法性
                if not is_valid_image_path(image_path):
                    print(f"跳过非法路径: 第 {line_number} 行, 路径: '{image_path}'")
                    continue
                if not valid_label_pattern.match(label):
                    print(f"跳过非法标签: 第 {line_number} 行, 标签: '{label}'")
                    continue

                writer.writerow([image_path, label])  # 写入合法行
            except Exception as e:
                print(f"跳过异常行: 第 {line_number} 行, 错误: {e}")


def count_invalid_rows(input_csv):
    """
    检查文件中非法字符和异常行数量，帮助确认过滤效果
    """
    invalid_rows = 0
    with open(input_csv, 'r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader((line.replace('\0', '') for line in infile))
        next(reader)  # 跳过表头

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2 or not is_valid_image_path(row[0].strip()) or not valid_label_pattern.match(row[1].strip()):
                invalid_rows += 1

    return invalid_rows


# 输入和输出文件路径
input_csv = "CCPD/cropped/val_labels.csv"  # 输入文件路径
output_csv = "CCPD/cropped/val_labels_cleaned.csv"  # 输出文件路径

# 第一步：清理非法行
filter_invalid_rows(input_csv, output_csv)
print(f"处理完成！合法行已保存至 {output_csv}")

# 第二步：统计清理后是否仍有非法行
remaining_invalid_rows = count_invalid_rows(output_csv)
print(f"清理完成后，仍存在 {remaining_invalid_rows} 条非法行。")