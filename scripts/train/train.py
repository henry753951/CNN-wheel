import os
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import questionary
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn_module.cuda.base import Conv2d as Conv2dBase

# from cnn_module.cuda.fft import Conv2d as Conv2dFft
# from cnn_module.cuda.img2col import Conv2d as Conv2dImg2Col
from models.model import ImageClassifierModel

AVAILABLE_CNNs = {
    "Official PyTorch": {"class": nn.Conv2d, "short_name": "official"},
    "Cuda Base": {"class": Conv2dBase, "short_name": "cuda_base"},
    # "Cuda FFT": {"class": Conv2dFft, "short_name": "cuda_fft"},
    # "Cuda Img2Col": {"class": Conv2dImg2Col, "short_name": "cuda_img2col"},
}

HYPER_PARAMETERS = {
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 10,
    "seed": 42,
    "val_split": 0.2,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_user_choices():
    if not AVAILABLE_CNNs:
        print("錯誤：沒有可用的 CNN！")
        exit()
    selected_cnn_name = questionary.select(
        "請選擇要使用的 CNN",
        choices=list(AVAILABLE_CNNs.keys()),
    ).ask()
    if selected_cnn_name is None:
        exit()
    cnn_short_name = AVAILABLE_CNNs.get(selected_cnn_name, {}).get("short_name", "unknown")
    timestamp = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M")
    default_path = f"models/bins/{cnn_short_name}/{timestamp}.pt"
    output_path = questionary.text("請輸入模型儲存路徑：", default=default_path).ask()
    if output_path is None:
        print("操作已取消。")
        exit()
    return selected_cnn_name, output_path


def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device
):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    pbar = tqdm(dataloader, desc="🚀 Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples * 100:.2f}%")
    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    pbar = tqdm(dataloader, desc="🧐 Validating", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix(
                loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples * 100:.2f}%"
            )
    return total_loss / total_samples, total_correct / total_samples


def train(conv_layer_class: type[nn.Module], save_path: str):
    set_seed(HYPER_PARAMETERS["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\t將使用裝置: {device}")

    # Set hyperparameters
    learning_rate = HYPER_PARAMETERS["learning_rate"]
    batch_size = HYPER_PARAMETERS["batch_size"]
    epochs = HYPER_PARAMETERS["epochs"]
    print(f"\tlearning_rate={learning_rate}, batch_size={batch_size}, epochs={epochs}")

    # Prepare dataset and dataloader
    print("📦 正在準備 CIFAR-10 資料集...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    val_size = int(len(full_train_dataset) * HYPER_PARAMETERS["val_split"])
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(HYPER_PARAMETERS["seed"])
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"\t{len(train_dataset)} 筆訓練資料, {len(val_dataset)} 筆驗證資料")
    print("📦 資料集準備完成！")

    # Create model, loss function, and optimizer
    model = ImageClassifierModel(conv_layer_class=conv_layer_class, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1} 結果: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%"
        )
    print("\n🚀 訓練完成！")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"⭐ 模型已成功儲存至: {save_path}")


if __name__ == "__main__":
    cnn_name, model_save_path = get_user_choices()
    selected_conv_class = AVAILABLE_CNNs[cnn_name]["class"]
    print(f"\n🚀 開始訓練，使用: {cnn_name}")
    train(selected_conv_class, model_save_path)
