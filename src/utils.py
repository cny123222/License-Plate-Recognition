import cv2
import matplotlib.pyplot as plt


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