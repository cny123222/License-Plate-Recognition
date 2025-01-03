import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# 数据集加载类
class CharacterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # 加载图像数据
        for label, char_folder in enumerate(os.listdir(data_dir)):
            folder_path = os.path.join(data_dir, char_folder)
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                self.data.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, label

# LSTM 模型定义
class LSTMRecognitionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMRecognitionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), x.size(2), -1)  # Flatten image into sequence
        out, _ = self.lstm(x)  # LSTM output
        out = out[:, -1, :]  # Use the last hidden state
        out = self.fc(out)
        return out

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # 计算训练准确率
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")

# 测试函数
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

# 数据路径和配置
chinese_data_dir = "path_to_chinese_data"  # 中文字符数据集路径
english_data_dir = "path_to_english_data"  # 英文与数字字符数据集路径

batch_size = 32
num_epochs = 10
input_dim = 28  # 图像宽度
hidden_dim = 128
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 加载中文数据集
chinese_dataset = CharacterDataset(chinese_data_dir, transform)
chinese_loader = DataLoader(chinese_dataset, batch_size=batch_size, shuffle=True)

# 加载英文与数字数据集
english_dataset = CharacterDataset(english_data_dir, transform)
english_loader = DataLoader(english_dataset, batch_size=batch_size, shuffle=True)

# 定义中文字符模型
num_chinese_classes = len(os.listdir(chinese_data_dir))
chinese_model = LSTMRecognitionModel(input_dim, hidden_dim, num_layers, num_chinese_classes)
chinese_criterion = nn.CrossEntropyLoss()
chinese_optimizer = optim.Adam(chinese_model.parameters(), lr=0.001)

# 定义英文与数字字符模型
num_english_classes = len(os.listdir(english_data_dir))
english_model = LSTMRecognitionModel(input_dim, hidden_dim, num_layers, num_english_classes)
english_criterion = nn.CrossEntropyLoss()
english_optimizer = optim.Adam(english_model.parameters(), lr=0.001)

# 训练中文字符模型
print("Training Chinese Character Model...")
train_model(chinese_model, chinese_loader, chinese_criterion, chinese_optimizer, device, num_epochs)

# 训练英文与数字字符模型
print("Training English and Numbers Model...")
train_model(english_model, english_loader, english_criterion, english_optimizer, device, num_epochs)

# 测试中文字符模型
print("Evaluating Chinese Character Model...")
evaluate_model(chinese_model, chinese_loader, device)

# 测试英文与数字字符模型
print("Evaluating English and Numbers Model...")
evaluate_model(english_model, english_loader, device)

