import torch
from torch import nn


class ImageClassifierModel(nn.Module):
    def __init__(self, conv_layer_class: tuple[type[nn.Module], dict], num_classes: int = 10):
        super().__init__()
        self.conv_layer_class = conv_layer_class

        # --- Block 1 ---
        self.conv1 = self.conv_layer_class[0](
            in_channels=3, out_channels=32, kernel_size=3, padding=1, **self.conv_layer_class[1]
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Block 2 ---
        self.conv2 = self.conv_layer_class[0](
            in_channels=32, out_channels=64, kernel_size=3, padding=1, **self.conv_layer_class[1]
        )
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


class ImageClassifierModelV2(nn.Module):
    def __init__(self, conv_layer_class: type[nn.Module], num_classes: int = 10):
        super().__init__()
        self.conv_layer_class = conv_layer_class

        # --- Block 1 ---
        # 3x32x32 -> 64x32x32
        self.conv1_1 = self.conv_layer_class(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = self.conv_layer_class(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64x16x16

        # --- Block 2 ---
        # 64x16x16 -> 128x16x16
        self.conv2_1 = self.conv_layer_class(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = self.conv_layer_class(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 128x8x8

        # --- Block 3 ---
        # 128x8x8 -> 256x8x8
        self.conv3_1 = self.conv_layer_class(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = self.conv_layer_class(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256x4x4

        # --- Dropout Layer ---
        self.dropout = nn.Dropout(0.5)

        # --- Fully Connected Layers ---
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256 * 4 * 4, out_features=512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Block 1 ---
        x = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)

        # --- Block 2 ---
        x = self.relu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)

        # --- Block 3 ---
        x = self.relu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.relu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)

        # --- Classifier ---
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)  # 輸出 是 logits，還沒 softmax
        return x
