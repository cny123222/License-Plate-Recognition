import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 输入为灰度图像
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 高度减半 (32 -> 16)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 高度减半 (16 -> 8)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # 高度减半 (8 -> 4)

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 新增卷积层
            nn.ReLU(),
            nn.MaxPool2d((4, 1), (4, 1), padding=(0, 0))  # 高度从 4 -> 1
        )

        # 独立的 LSTM 层
        self.lstm1 = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # 卷积特征提取
        conv_out = self.cnn(x)
        b, c, h, w = conv_out.size()
        assert h == 1, f"Feature height must be 1, but got {h}."
        conv_out = conv_out.squeeze(2).permute(0, 2, 1)  # (B, W, C)

        # 第一个 LSTM 层
        rnn_out, _ = self.lstm1(conv_out)

        # 第二个 LSTM 层
        rnn_out, _ = self.lstm2(rnn_out)

        # 全连接层
        output = self.fc(rnn_out)
        return output