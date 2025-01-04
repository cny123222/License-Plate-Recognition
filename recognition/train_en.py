import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from tqdm import tqdm

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 将单通道扩展为三通道
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 设置设备
if torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
dataset = datasets.ImageFolder(root='dataset/chars2', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 定义 ResNet18 模型
class CharClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CharClassifier, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 模型实例化
num_classes_char2 = len(dataset.classes)  # 根据文件夹自动检测类别数量
model_char2 = CharClassifier(num_classes=num_classes_char2).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_char2.parameters(), lr=0.001, weight_decay=1e-4)

# 训练循环
for epoch in range(10):  # 可调整训练轮数
    model_char2.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{10}", leave=False)
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model_char2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# 保存模型
torch.save(model_char2.state_dict(), "eng_num_char_classifier.pth")
print("Model saved as eng_num_char_classifier.pth")