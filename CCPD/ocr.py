import os
import cv2
import pandas as pd
from paddleocr import PaddleOCR
from tqdm import tqdm

# PaddleOCR 初始化
ocr = PaddleOCR(use_angle_cls=False, lang='ch')  # 使用中文模型

def preprocess_image(image_path):
    """
    对图像进行预处理（可选）
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
    if image is None:
        raise ValueError(f"无法读取图像 {image_path}")
    image = cv2.resize(image, (256, 64), interpolation=cv2.INTER_CUBIC)  # 调整大小
    return image

def evaluate_paddleocr(csv_file):
    """
    使用 PaddleOCR 对车牌进行识别并评估准确率
    """
    data = pd.read_csv(csv_file, encoding='utf-8').head(10)
    total_samples = len(data)
    correct_predictions = 0
    errors = []

    for _, row in tqdm(data.iterrows(), total=total_samples, desc="Evaluating OCR"):
        image_path, label = row['image_path'], row['label']

        try:
            # 图像预处理
            image = preprocess_image(image_path)

            # OCR 识别
            results = ocr.ocr(image_path)
            if results and results[0]:
                ocr_result = results[0][0][1][0]  # 取第一个检测框的结果
            else:
                ocr_result = ""

            # 比较结果
            if ocr_result == label:
                correct_predictions += 1
            else:
                errors.append((image_path, label, ocr_result))

        except Exception as e:
            errors.append((image_path, label, f"ERROR: {e}"))

    accuracy = correct_predictions / total_samples
    print(f"总样本数: {total_samples}")
    print(f"OCR 准确识别数: {correct_predictions}")
    print(f"OCR 准确率: {accuracy:.4f}")

    # 输出错误案例
    print("\n错误案例:")
    for image_path, label, ocr_result in errors[:10]:  # 打印前10个错误
        print(f"图像路径: {image_path}, 实际标签: {label}, OCR 结果: {ocr_result}")

    return accuracy, errors

if __name__ == "__main__":
    # 指定数据集的 CSV 文件路径
    csv_file = "CCPD/cropped/train_labels_cleaned.csv"

    # 评估 PaddleOCR
    evaluate_paddleocr(csv_file)