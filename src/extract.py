import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from utils import cv_show, cv_save


def extract_plate(origin_image):
    """
    提取车牌及颜色
    """
    item, _ = extract(origin_image)
    # img_rotated = rotation(item, origin_image)
    # _, result = extract(img_rotated)
    
    # # 检测车牌颜色
    # hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    # plate_color = detect_plate_color(hsv_result)
    
    # return result, plate_color

    return _, "Blue"


def gray_guss(image):
    """
    高斯模糊去噪
    """
    image = cv2.GaussianBlur(image, (3, 3), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image


def is_plate_color(hsv_img, lower_bound, upper_bound):
    """
    车牌颜色过滤
    """
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return cv2.countNonZero(mask) > 0


def detect_plate_color(hsv_img):
    """
    检测车牌颜色
    """
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    if is_plate_color(hsv_img, lower_blue, upper_blue):
        return "Blue"
    elif is_plate_color(hsv_img, lower_yellow, upper_yellow):
        return "Yellow"
    elif is_plate_color(hsv_img, lower_green, upper_green):
        return "Green"
    else:
        return "Unknown"


def rotation(contours, image):
    """
    旋转图像
    """
    cnt = contours
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # 确保vx和vy是标量值
    k = vy[0] / vx[0] if vx[0] != 0 else float('inf')  # 防止除以0
    angle = math.atan(k)
    angle = math.degrees(angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.8)
    img = cv2.warpAffine(image, M, (int(w * 1.1), int(h * 1.1)))
    return img


def extract(origin_image):
    """
    提取车牌
    """
    # 读取图像
    image = origin_image.copy()

    # 高斯模糊去噪得到去噪后的灰度图
    gray_image = gray_guss(image)

    cv_save("figures/1gray_image.png", gray_image)

    # 使用Sobel在x方向上进行边缘检测
    Sobel_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0)
    absX = cv2.convertScaleAbs(Sobel_x)
    image = absX

    cv_save("figures/2Sobel_x.png", image)

    # 阈值化处理
    ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    cv_save("figures/3threshold.png", image)

    # 白化，参数和迭代次数需要调整
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernelX, iterations=3)

    cv_save("figures/4morphologyEx.png", image)

    # 形态学操作
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    image = cv2.dilate(image, kernelX)
    image = cv2.erode(image, kernelX)
    image = cv2.erode(image, kernelY)
    image = cv2.dilate(image, kernelY)
    image = cv2.medianBlur(image, 21)

    cv_save("figures/5morphologyEx2.png", image)

    # 轮廓检测
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image1 = origin_image.copy()
    cv2.drawContours(image1, contours, -1, (0, 255, 0), 5)

    cv_save("figures/6drawContours.png", image1)

    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        width = rect[2]
        height = rect[3]
        # 根据轮廓的形状特点和颜色，确定车牌的轮廓位置并截取图像
        if (width > (height * 2.5)) and (width < (height * 5)):
            image = origin_image[y:y + height, x:x + width]

            cv_save("figures/8plate.png", image)

            return item, image
            

if __name__ == '__main__':
    origin_image = cv2.imread("test_images/002.png")
    plate_image, plate_color = extract_plate(origin_image)

    print(f"车牌颜色: {plate_color}")