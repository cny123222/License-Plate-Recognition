import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import wandb

# 初始化 WandB
wandb.init(project="char-classification", config={
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "scheduler_step": 10,
    "scheduler_gamma": 0.1
})
config = wandb.config

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

# 加载数据集并划分为训练集和测试集
dataset = datasets.ImageFolder(root='dataset/chars2', transform=transform)
num_classes = len(dataset.classes)
print(f"Number of classes: {num_classes}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

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

# 初始化模型
model = CharClassifier(num_classes=num_classes).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

final_test_accuracy = 0.0

# 训练过程
for epoch in range(config.epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    
    # 训练集准确率
    train_accuracy = evaluate(model, train_loader, device)
    # 测试集准确率
    test_accuracy = evaluate(model, test_loader, device)
    final_test_accuracy = test_accuracy
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # 记录到 WandB
    wandb.log({"epoch": epoch + 1, "loss": total_loss, "train_accuracy": train_accuracy, "test_accuracy": test_accuracy})

# 保存模型
model_filename = f"checkpoints/char_classifier_bs{config.batch_size}_lr{config.learning_rate}_wd{config.weight_decay}_acc{final_test_accuracy:.4f}.pth"
torch.save(model.state_dict(), model_filename)
wandb.save(model_filename)
print(f"Model saved as {model_filename}")