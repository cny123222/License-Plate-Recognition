import cv2
import numpy as np


def detect_plate_color(image):
    # 转换为HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围（HSV范围）
    # 黄色: H(20-30), S(100-255), V(100-255)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # 蓝色: H(100-130), S(100-255), V(100-255)
    blue_lower = np.array([100, 100, 100])
    blue_upper = np.array([130, 255, 255])

    # 绿色: H(40-80), S(100-255), V(100-255)
    green_lower = np.array([40, 100, 100])
    green_upper = np.array([80, 255, 255])

    # 创建掩码
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # 统计每种颜色的像素数量
    yellow_count = cv2.countNonZero(yellow_mask)
    blue_count = cv2.countNonZero(blue_mask)
    green_count = cv2.countNonZero(green_mask)

    # 比较颜色数量，找出主色
    color_counts = {'Yellow': yellow_count, 'Blue': blue_count, 'Green': green_count}

    # 找出像素最多的颜色
    dominant_color = max(color_counts, key=color_counts.get)
    print(dominant_color)
    return dominant_color

