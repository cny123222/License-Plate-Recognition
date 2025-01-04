import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义中文字符识别模型
class ChineseCharClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ChineseCharClassifier, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 定义英文和数字字符识别模型
class EngNumCharClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EngNumCharClassifier, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 加载中文字符识别模型
num_classes_chinese = 31
chinese_model = ChineseCharClassifier(num_classes=num_classes_chinese)
chinese_model.load_state_dict(torch.load("chinese_char_classifier.pth"))
chinese_model.eval()

# 加载英文和数字字符识别模型
num_classes_eng_num = 34
eng_num_model = EngNumCharClassifier(num_classes=num_classes_eng_num)
eng_num_model.load_state_dict(torch.load("eng_num_char_classifier.pth"))
eng_num_model.eval()

# 类别映射
chinese_class_map = [
    '川', '鄂', '赣', '甘', '贵', '桂', '黑', '沪', '冀', '津', '京', '吉', '辽', '鲁', '蒙', 
    '闽', '宁', '青', '琼', '陕', '苏', '晋', '皖', '湘', '新', '豫', '渝', '粤', '云', '藏', '浙' 
]
eng_num_class_map = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
]


def preprocess_image(image):
    """
    输入：单个字符图像 (PIL Image 或 numpy.ndarray)
    输出：预处理后的张量
    """
    if isinstance(image, np.ndarray):  # 如果是 numpy.ndarray，转换为 PIL.Image
        image = Image.fromarray(image)
    return transform(image).unsqueeze(0)  # 增加 batch 维度


def recognize_characters(char_images):
    """
    输入：字符图像列表
    输出：识别结果字符串
    """
    if len(char_images) < 5:
        return "字符数不足 5, 识别错误"

    result = []

    # 处理第一位字符（中文）
    first_tensor = preprocess_image(char_images[0])
    with torch.no_grad():
        output = chinese_model(first_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        result.append(chinese_class_map[predicted_class])

    # 处理后续字符（英文和数字）
    for char_image in char_images[1:]:
        char_tensor = preprocess_image(char_image)
        with torch.no_grad():
            output = eng_num_model(char_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            result.append(eng_num_class_map[predicted_class])

    return ''.join(result)