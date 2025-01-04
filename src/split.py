import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image):
    '''
    灰度化处理
    image: 车牌图像
    - return  gray_iamge : 灰度图像
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def binarize_image(image):
    '''
    二值化处理
    image: 灰度图像
    - return binary_image: 二值化图像
    '''
    
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def detect_character_color(binary_image):
    """
    判断二值化图像中字符的颜色。
    - 返回 'white' 表示背景为白色。
    - 返回 'black' 表示背景为黑色。
    """
    # 计算图像中黑色和白色像素的数量
    white_pixel_count = cv2.countNonZero(binary_image)  # 非零像素为白色
    black_pixel_count = binary_image.size - white_pixel_count  # 总像素减去白色像素为黑色

    # 判断字符颜色
    if white_pixel_count > black_pixel_count:
        return 'white'  # 字符为白色，背景为黑色
    else:
        return 'black'  # 字符为黑色，背景为白色


def connect_characters(binary_image):
    """
    使用形态学膨胀操作连接字符内部的断裂区域。
    binary_image: 二值化图像
    - return connected_image: 连接后的二值化图像
    """
    # 定义膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # 膨胀操作
    connected_image = cv2.dilate(binary_image, kernel, iterations=1)

    return connected_image


def connect_characters_double(image):
    '''
    具体功能同上，但只作用于前1/8的区域，用于连接汉字的断裂区域
    image: 二值化图像
    - return result_image: 连接后的二值化图像
    '''
    # 获取图像尺寸
    height, width = image.shape

    # 定义前 1/8 的区域
    end_row = width // 8  # 计算前 1/8 的行数
    roi = image[0:height, 0:end_row]  # 提取前 1/8 区域

    # 创建膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # 对 ROI 执行膨胀
    dilated_roi = cv2.dilate(roi, kernel, iterations=1)

    # 将膨胀结果合并回原图
    result_image = image.copy()
    result_image[0:height, 0:end_row] = dilated_roi

    return result_image


def horizontal_projection(binary_image,bi_image):
    '''
    水平投影，判断字符的具体范围(上下边界)
    binary_image:二值化图像
    bi_image: 二值化图像(未经膨胀)
    - return horizontal_area: 具体范围的图像
    - return image_for_split: 用于分割的图像
    '''

    # 水平方向投影：计算每一行的像素和
    horizontal_sum = np.sum(binary_image, axis=1)

    # 使用水平投影来分割字符区域
    horizontal_threshold = max(horizontal_sum) // 6  # 设置水平分割的阈值
    horizontal_split_positions = find_split_positions_horizontal(horizontal_sum, horizontal_threshold)
    
    start, end = max(horizontal_split_positions, key=lambda x: x[1] - x[0])

    horizontal_area = binary_image[start:end+1, :]
    image_for_split = bi_image[start:end+1,:]
    return horizontal_area,image_for_split



def vertical_projection(binary_image):
    '''
    垂直方向上的投影，用于判断字符之间的分割位置
    binary_image : 经过水平投影之后的二值化图像
    - return vertical_sum : 垂直方向上的投影
    '''

    # 垂直方向投影：计算每一列的像素和
    vertical_sum = np.sum(binary_image, axis=0)
    return vertical_sum


# 找到字符的分割位置
def find_split_positions_horizontal(projection, threshold):
    ''' 
    找到字符的具体位置，上下确定边界
    projection: 水平投影值
    thresohld : 阈值
    - return split_positions : 字符的具体范围（上下）
    '''

    split_positions = []
    start = None
    for i in range(len(projection)):
        if projection[i] > threshold:
            if start is None:
                start = i  # 记录字符开始的点
        else:
            if start is not None:
                split_positions.append((start, i - 1))
                start = None
    # 如果最后还有一个字符
    if start is not None:
        split_positions.append((start, len(projection) - 1))

    return split_positions


def find_split_positions_vertical(vertical_proj):
    """
    进行初步分割，找出字符区域（峰值）和字符间的空隙（谷值）
    vertical_proj : 垂直投影
    - return character_regions : 初步分割后的各字符区域
    """
    threshold = np.max(vertical_proj) * 0.2  # 投影值的20%作为分割阈值
    in_character = False
    character_regions = []
    start = 0

    for i, value in enumerate(vertical_proj):
        if value > threshold and not in_character:  # 检测字符区域开始
            in_character = True
            start = i
        elif value <= threshold and in_character:  # 检测字符区域结束
            in_character = False
            character_regions.append((start, i - 1))
    
    return character_regions


def calculate_max_center_distance(character_regions):
    """
    根据初步分割，计算字符间的最大中心距
    character_regions : 初步分割后的歌字符区域
    - return max_distance : 最大中心距
    """
    centers = [(start + end) // 2 for start, end in character_regions]  # 每个字符区域的中心点
    max_distance = 0

    for i in range(len(centers) - 1):
        distance = centers[i + 1] - centers[i]  # 相邻字符中心点距离
        max_distance = max(max_distance, distance)
    
    return max_distance


def vertical_segmentation(bi_image,image, max_distance):
    """
    根据最大中心距分割字符
    bi_iamge : 用于显示分割结果的二值化图像（未经膨胀）
    image : 二值化图像(经过膨胀)
    max_distance : 最大中心距
    - return segments : 分割后的字符图像集 
    """

    vertical_proj = vertical_projection(image)

    # print(max_distance,'max_distance')
    threshold = np.max(vertical_proj) * 0.1  # 分割阈值为最大中心距的一半
    in_character = False
    start = 0
    segments = []

    for i, value in enumerate(vertical_proj):
        if value > threshold and not in_character:  # 检测字符开始
            in_character = True
            start = i
            # print('start',start)
        elif value <= threshold and in_character:  # 检测字符结束
            in_character = False

            if(i - start > max_distance // 7) and np.max(vertical_proj[start:i]) > 5 * threshold:
                segments.append(bi_image[:, start:i])  # 截取字符块
    if in_character and len(vertical_proj) - start > max_distance // 3:

        segments.append(bi_image[:, start:i])

    return segments


def show_segmented_characters(characters):
    '''
    显示分割结果
    characters : 分割后的字符图像集
    '''
    for idx, char in enumerate(characters):
        plt.subplot(1, len(characters), idx+1)
        plt.imshow(char, cmap='gray')
        plt.axis('off')
    plt.show()


def split_char(image):
    '''
    主函数，运行时只需调用该函数即可
    image : 车牌图像
    - return segments : 分割后的字符图像集
    '''

    # 1. 加载图像并进行二值化
    gray_image = load_image(image)
    bi_image = binarize_image(gray_image)

    # 2. 判断字体颜色，若为黑色，则翻转颜色
    if(detect_character_color(bi_image) == 'white'):
        bi_image = cv2.bitwise_not(bi_image)  # 反转颜色

    # 3. 膨胀化，连接断裂区域
    binary_image = connect_characters(bi_image)

    # 4. 再次膨胀化，应用于汉字断裂区域
    binary_image = connect_characters_double(binary_image)

    # 5. 水平投影 + 分割
    horizontal_proj,bi_image = horizontal_projection(binary_image,bi_image)

    # 6. 垂直投影分割
    vertical_proj = vertical_projection(horizontal_proj)

    # 7. 初步分割，确定最大中心距
    character_regions = find_split_positions_vertical(vertical_proj)
    max_distance = calculate_max_center_distance(character_regions)

    # 8. 分割字符
    segments = vertical_segmentation(bi_image,horizontal_proj, max_distance)

    #  显示分割后的字符
    show_segmented_characters(segments)

    return segments
