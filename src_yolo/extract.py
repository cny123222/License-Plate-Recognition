import yolov5
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import traverse_images, cv_show

# 加载 YOLOv5 模型
model = yolov5.load('keremberke/yolov5n-license-plate') 

# 配置 YOLOv5 模型参数
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image


def extract_plate_yolo(image, expand_ratio=0.2):
    """
    使用 YOLOv5 模型定位车牌并提取颜色，同时对检测框进行扩展
    :param image: 输入的原始图像
    :param expand_ratio: 扩展比例，每边扩展的比例
    :return: 检测到的车牌图像及车牌颜色
    """
    # YOLOv5 模型推理
    results = model(image, size=640)
    predictions = results.pred[0]

    # 如果没有检测到任何车牌
    if len(predictions) == 0:
        return None, "Unknown"

    # 获取第一个检测到的车牌边界框（默认置信度最高）
    box = predictions[0, :4].cpu().numpy().astype(int)  # 转换为整数坐标
    x1, y1, x2, y2 = expand_bbox(image, box, expand_ratio)

    # 仿射变换矫正车牌倾斜
    plate_image = image[y1:y2, x1:x2]

    # 提取角点并进行倾斜矫正
    plate_image_corrected = correct_skew(plate_image)

    # 进行颜色检测
    plate_color = detect_plate_color(plate_image_corrected)

    return plate_image_corrected, plate_color


def expand_bbox(image, box, scale=0.2):
    """
    扩展 YOLO 矩形框
    :param image: 原始图像
    :param box: YOLO 的矩形框 [x1, y1, x2, y2]
    :param scale: 扩展比例（每边扩展的比例）
    :return: 扩展后的矩形框
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # 计算扩展大小
    dx = int((x2 - x1) * scale)
    dy = int((y2 - y1) * scale)

    # 应用扩展，并确保边界有效
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(width, x2 + dx)
    y2 = min(height, y2 + dy)

    return [x1, y1, x2, y2]



def correct_skew(image):
    """
    矫正车牌倾斜
    :param image: 输入的车牌图像
    :return: 矫正后的车牌图像
    """
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cv_show("Edged", edged)

    # 寻找轮廓
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # 如果没有找到轮廓，返回原图
    
    cv_show("Contours", cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2))

    # 找到面积最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    cv_show("Largest Contour", cv2.drawContours(image.copy(), [largest_contour], -1, (0, 255, 0), 2))

    # 最小外接矩形
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv_show("Rotated Box", cv2.drawContours(image.copy(), [box], -1, (0, 255, 0), 2))

    # 计算仿射变换矩阵
    width = int(rect[1][0])
    height = int(rect[1][1])

    if width > height:  # 确保宽大于高
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
    else:  # 宽和高调换
        dst_pts = np.array([[0, width - 1],
                            [0, 0],
                            [height - 1, 0],
                            [height - 1, width - 1]], dtype="float32")

    src_pts = box.astype("float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (max(width, height), min(width, height)))

    return warped


def correct_skew_hough(image):
    """
    使用 Hough 变换矫正车牌倾斜
    :param image: 输入的车牌图像
    :return: 矫正后的车牌图像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    cv_show("Edged", edged)

    # 使用 Hough 变换检测直线
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 100)
    if lines is None:
        return image  # 如果没有检测到直线，返回原图
    
    cv_show("Hough Lines", cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR))

    # 计算主要线条的角度
    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta * 180 / np.pi) - 90  # 转为角度
        angles.append(angle)

    # 计算平均倾斜角度
    avg_angle = np.mean(angles)

    # 旋转图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated


def detect_plate_color(image):
    """
    检测车牌颜色
    :param image: 车牌图像
    :return: 车牌颜色 ("Blue", "Yellow", "Green", "Unknown")
    """
    # 转换为 HSV 色彩空间
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # 检测颜色
    if is_plate_color(hsv_img, lower_blue, upper_blue):
        return "Blue"
    elif is_plate_color(hsv_img, lower_yellow, upper_yellow):
        return "Yellow"
    elif is_plate_color(hsv_img, lower_green, upper_green):
        return "Green"
    else:
        return "Unknown"


def is_plate_color(hsv_img, lower_bound, upper_bound):
    """
    判断图像是否为指定颜色
    :param hsv_img: HSV 格式的图像
    :param lower_bound: HSV 颜色下界
    :param upper_bound: HSV 颜色上界
    :return: 是否为指定颜色
    """
    mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    return cv2.countNonZero(mask) > 0


# 测试代码
if __name__ == "__main__":
    image_paths = traverse_images("test_images")
    for image_path in image_paths:
        # 加载测试图像
        test_image = cv2.imread(image_path)

        # 使用 YOLOv5 提取车牌
        plate_image, plate_color = extract_plate_yolo(test_image, expand_ratio=0.1)

        # 显示结果
        if plate_image is not None:
            print(f"车牌颜色: {plate_color}")
            cv2.imshow("Detected Plate", plate_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("未检测到车牌")