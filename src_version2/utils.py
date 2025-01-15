import cv2
import matplotlib.pyplot as plt
import os


def cv_show(name, img):
    """
    图片显示
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plt_show_color(img):
    """
    彩色图片显示
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def plt_show_gray(img):
    """
    灰度图片显示
    """
    plt.imshow(img, cmap='gray')
    plt.show()


def traverse_images(directory):
    """
    遍历目录中的所有图片文件，包括子文件夹
    :param directory: 要遍历的目录路径
    """
    # 定义支持的图片扩展名
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    image_paths = []

    # 遍历目录及其子目录
    for root, _, files in os.walk(directory):
        for file in files:
            # 获取文件的扩展名并检查是否为有效图片
            if os.path.splitext(file)[-1].lower() in valid_extensions:
                # 拼接完整路径
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    
    return image_paths
