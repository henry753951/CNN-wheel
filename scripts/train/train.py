import json
import os
import random
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import questionary
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn_methods import AVAILABLE_CNNs
from models.model import ImageClassifierModel, ImageClassifierModelV2

HYPER_PARAMETERS = {
    "learning_rate": {"initial_lr": 1e-3, "min_lr": 1e-5, "factor": 0.5, "patience": 2},
    "batch_size": 64,
    "epochs": 10,
    "seed": 42,
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
    args = AVAILABLE_CNNs.get(selected_cnn_name, {}).get("args", {})
    timestamp = datetime.now(tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M")
    default_path = f"models/bins/{cnn_short_name}/{timestamp}.pt"
    output_path = questionary.text("請輸入模型儲存路徑：", default=default_path).ask()
    if output_path is None:
        print("操作已取消。")
        exit()
    return (selected_cnn_name, output_path, args)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    start_time = time.time()

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

    end_time = time.time()
    epoch_duration = end_time - start_time

    return total_loss / total_samples, total_correct / total_samples, epoch_duration


def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    start_time = time.time()

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

    end_time = time.time()
    epoch_duration = end_time - start_time

    return total_loss / total_samples, total_correct / total_samples, epoch_duration


def train(conv_layer_class: tuple[type[nn.Module], dict], save_path: str, is_cpu: bool = False):
    set_seed(HYPER_PARAMETERS["seed"])

    # Set hyperparameters
    initial_lr = HYPER_PARAMETERS["learning_rate"]["initial_lr"]
    min_lr = HYPER_PARAMETERS["learning_rate"]["min_lr"]
    factor = HYPER_PARAMETERS["learning_rate"]["factor"]
    patience = HYPER_PARAMETERS["learning_rate"]["patience"]
    batch_size = HYPER_PARAMETERS["batch_size"]
    epochs = HYPER_PARAMETERS["epochs"]
    print(
        f"\tinitial_lr={initial_lr}, min_lr={min_lr}, factor={factor}, patience={patience}, batch_size={batch_size}, epochs={epochs}"
    )

    # Prepare dataset and dataloader
    print("📦 正在準備 CIFAR-10 資料集...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"\t{len(train_dataset)} 筆訓練資料, {len(val_dataset)} 筆驗證資料")
    print("📦 資料集準備完成！")

    # Create model, loss function, optimizer, and scheduler

    if is_cpu:
        print("🔄 將模型轉換為 CPU 模式...")
        device = torch.device("cpu")
    else:
        print("🔄 將模型轉換為 CUDA 模式...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImageClassifierModel(conv_layer_class=conv_layer_class, num_classes=10).to(device)
    print(f"\t將使用裝置: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )

    # Initialize metrics storage
    metrics = {"hyperparameters": HYPER_PARAMETERS, "epochs": []}

    # Train the model
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_time = validate_one_epoch(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1} 結果: \n"
            f"    Train -> Loss: {train_loss:.4f}, Acc: {train_acc * 100:.2f}%, Time: {train_time:.2f}s\n"
            f"    Valid -> Loss: {val_loss:.4f}, Acc: {val_acc * 100:.2f}%, Time: {val_time:.2f}s\n"
            f"    Learning Rate: {current_lr:.6f}"
        )

        # Store metrics for this epoch
        metrics["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_time": train_time,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_time": val_time,
                "learning_rate": current_lr,
            }
        )

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

    print("\n🚀 訓練完成！")

    # Save the model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"⭐ 模型已成功儲存至: {save_path}")

    # Save metrics to JSON
    metrics_path = os.path.splitext(save_path)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"📊 訓練數據已儲存至: {metrics_path}")


if __name__ == "__main__":
    cnn_name, model_save_path, args = get_user_choices()
    selected_conv_class = AVAILABLE_CNNs[cnn_name]["class"]
    print(f"\n🚀 開始訓練，使用: {cnn_name}")
    train((selected_conv_class, args), model_save_path, is_cpu="CPU" in cnn_name)
