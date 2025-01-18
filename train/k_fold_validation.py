import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import numpy as np

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

if torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载完整数据集
dataset = datasets.ImageFolder(root='dataset/chars2', transform=transform)
data_labels = [y for _, y in dataset.imgs]  # 提取每张图片的标签

# 定义 K 折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
num_classes = len(dataset.classes)
print(f"Number of classes: {num_classes}")

# 模型定义
class CharClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CharClassifier, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 测试函数
def evaluate(model, data_loader, device):
    model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            total_preds.extend(preds.cpu().numpy())
            total_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(total_labels, total_preds)
    return accuracy

# K 折交叉验证
fold_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.imgs, data_labels)):
    print(f"Fold {fold + 1}")
    
    # 数据划分
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = CharClassifier(num_classes=num_classes).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)
    
    # 训练过程
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
    # 验证集评估
    val_accuracy = evaluate(model, val_loader, device)
    print(f"Fold {fold + 1}, Validation Accuracy: {val_accuracy:.4f}")
    fold_accuracies.append(val_accuracy)

# 平均性能
print(f"Average Validation Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")