from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import wandb  # 引入 wandb
from crnn import CRNN
from tqdm import tqdm

# 初始化 wandb
wandb.init(
    project="license-plate-recognition",
    config={
        "epochs": 32,
        "batch_size": 32,
        "learning_rate": 0.001,
        "loss": "CTCLoss",
    }
)

# 定义字符映射规则
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
             "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# 字符集
charset = provinces + ads
char_to_idx = {char: idx for idx, char in enumerate(charset)}
idx_to_char = {idx: char for idx, char in enumerate(charset)}

# 数据集类
class LicensePlateDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        img = Image.open(img_path).convert('L')  # 灰度图像
        if self.transform:
            img = self.transform(img)
        label_idx = [char_to_idx[c] for c in label]
        return img, torch.tensor(label_idx)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和验证集
train_dataset = LicensePlateDataset('CCPD/cropped/train_labels_cleaned.csv', transform=transform)
val_dataset = LicensePlateDataset('CCPD/cropped/val_labels_cleaned.csv', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# CTC Loss
ctc_loss = nn.CTCLoss(blank=len(charset))  # 最后一类作为空白符

# 模型、优化器和设备设置
if torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CRNN(len(charset) + 1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# 解码预测结果
def decode_predictions(predictions):
    results = []
    for pred in predictions:
        decoded = ""
        previous_char = None
        for char_idx in pred:
            if char_idx == len(charset) or char_idx == previous_char:  # 跳过空白符或重复字符
                continue
            decoded += idx_to_char[char_idx]
            previous_char = char_idx
        results.append(decoded)
    return results

# 计算验证集准确率
def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predictions = outputs.max(2)  # 获取预测结果
            predictions = predictions.transpose(1, 0).cpu().numpy()  # 转置为每个序列的形状
            decoded_preds = decode_predictions(predictions)
            for pred, label in zip(decoded_preds, labels):
                target_label = "".join([idx_to_char[idx.item()] for idx in label])
                if pred == target_label:
                    correct += 1
                total += 1
    return correct / total

# 训练循环
epochs = wandb.config.epochs

def train_one_epoch(model, train_loader, optimizer, ctc_loss, device):
    """
    训练一个 epoch
    """
    model.train()
    epoch_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)

        # 获取时间步长（T）和批量大小
        batch_size = outputs.size(0)  # 批量大小
        seq_len = outputs.size(1)     # 序列长度

        input_lengths = torch.full(
            size=(batch_size,),  # 批量大小
            fill_value=seq_len,  # 每个序列的长度
            dtype=torch.int32,
            device='cpu'
        )

        # 确保目标序列长度与批次中的样本一致
        target_lengths = torch.tensor(
            [len(label) for label in labels],
            dtype=torch.int32,
            device='cpu'
        )

        # 转换 outputs 的形状
        outputs = outputs.permute(1, 0, 2)  # 从 (N, T, C) 转为 (T, N, C)

        # 计算 CTC Loss
        loss = ctc_loss(outputs.log_softmax(2).cpu(), labels.cpu(), input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def evaluate_model(model, val_loader, device, print_samples=5):
    """
    在验证集上评估模型性能，并打印部分预测样本
    :param model: 模型
    :param val_loader: 验证集 DataLoader
    :param device: 使用的设备 (CPU/GPU)
    :param print_samples: 打印的预测样本数
    :return: 验证集准确率
    """
    model.eval()
    correct = 0
    total = 0
    printed = 0  # 已打印样本计数

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)

            # 获取预测结果
            outputs = outputs.permute(1, 0, 2)  # 转换为 (T, N, C)
            _, predictions = outputs.max(2)  # 获取类别索引
            predictions = predictions.transpose(1, 0).cpu().numpy()

            # 解码预测结果
            decoded_preds = decode_predictions(predictions)
            for pred, label in zip(decoded_preds, labels):
                target_label = "".join([idx_to_char[idx.item()] for idx in label])
                if pred == target_label:
                    correct += 1
                total += 1

                # 打印部分预测和真实标签
                if printed < print_samples:
                    print(f"预测: {pred}, 真实: {target_label}")
                    printed += 1

    accuracy = correct / total if total > 0 else 0
    print(f"验证集准确率: {accuracy:.4f}")
    return accuracy


for epoch in range(epochs):
    # 训练一个 epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, ctc_loss, device)

    # 在验证集上评估准确率
    val_accuracy = evaluate_model(model, val_loader, device)

    # 打印日志
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 使用 wandb 记录训练和验证过程
    wandb.log({"Epoch": epoch + 1, "Train Loss": train_loss, "Val Accuracy": val_accuracy})

# 保存模型
torch.save(model.state_dict(), "crnn_model.pth")
wandb.save("crnn_model.pth")