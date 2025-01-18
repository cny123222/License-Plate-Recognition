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
        return None

    # 获取第一个检测到的车牌边界框（默认置信度最高）
    box = predictions[0, :4].cpu().numpy().astype(int)  # 转换为整数坐标
    x1, y1, x2, y2 = expand_bbox(image, box, expand_ratio)
    plate_image = image[y1:y2, x1:x2]

    # 提取角点并进行倾斜矫正
    # plate_image_corrected = correct_skew(plate_image)
    # plate_image_corrected = locate_license_plate(plate_image)


    return plate_image


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


def locate_license_plate(image):
    """
    精确定位车牌位置
    :param image: 含有车牌的大致范围图像
    :return: 矫正后的车牌图像
    """
    # 1. 预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 100, 200)

    # cv_show("Edged", edged)

    # 形态学操作，闭操作连接边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # 2. 提取轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv_show("Closed", closed)

    # 3. 几何约束筛选车牌轮廓
    plate_contour = None
    for contour in contours:
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算面积和长宽比
        width = int(rect[1][0])
        height = int(rect[1][1])
        if width == 0 or height == 0:
            continue
        aspect_ratio = max(width, height) / min(width, height)

        # 计算轮廓面积与矩形面积比值
        area = cv2.contourArea(contour)
        rect_area = width * height
        if rect_area == 0:
            continue
        extent = area / rect_area

        # 几何约束判断
        if 2 < aspect_ratio < 5 and extent > 0.6 and 500 < rect_area < 50000:
            plate_contour = box
            break  # 假设只有一个车牌

    if plate_contour is None:
        print("未找到符合条件的车牌轮廓")
        return None
    
    cv_show("Plate Contour", cv2.drawContours(image.copy(), [plate_contour], -1, (0, 255, 0), 2))

    # 4. 透视变换矫正车牌
    rect = cv2.minAreaRect(plate_contour)
    width, height = int(rect[1][0]), int(rect[1][1])
    if width > height:
        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    else:
        dst_pts = np.array([[0, 0], [height - 1, 0], [height - 1, width - 1], [0, width - 1]], dtype="float32")

    src_pts = plate_contour.astype("float32")
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

    return dominant_color


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

'''
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
            print("未检测到车牌")'''

if __name__ == '__main__':
    origin_image = cv2.imread("../test_images/030.jpg")
    plate_image = extract_plate_yolo(origin_image)
    print(plate_image == None)
    # cv_show("plate", plate_image)
    cv2.imwrite(".tmp/plate.jpg", plate_image)