import torch
from torch import nn


class ImageClassifierModel(nn.Module):
    def __init__(self, conv_layer_class: type[nn.Module], num_classes: int = 10):
        super().__init__()
        self.conv_layer_class = conv_layer_class

        # --- Block 1 ---
        self.conv1 = self.conv_layer_class(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 ---
        self.conv2 = self.conv_layer_class(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dropout Layer ---
        self.dropout = nn.Dropout(0.4)

        # --- Fully Connected Layers ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout(x)

        x = self.flatten(x)

        x = self.relu3(self.fc1(x))
        x = self.fc2(x)  # 輸出 是 logits，還沒 softmax
        return x
