import os
import time
from pathlib import Path

import questionary
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from cnn_methods import AVAILABLE_CNNs
from models.model import ImageClassifierModel


def get_user_model_selection() -> tuple[str, type[nn.Module], dict]:
    selected_cnn_name = questionary.select(
        "請選擇要評估的 CNN 類型：",
        choices=list(AVAILABLE_CNNs.keys()),
    ).ask()

    if selected_cnn_name is None:
        print("操作已取消。")
        exit()

    cnn_info = AVAILABLE_CNNs[selected_cnn_name]
    conv_layer_class = cnn_info["class"]
    args = cnn_info.get("args", {})
    model_dir = Path(f"models/bins/{cnn_info['short_name']}")

    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"錯誤：在目錄 '{model_dir}' 中找不到任何已儲存的模型。")
        exit()

    saved_models = sorted(model_dir.glob("*.pt"), key=os.path.getmtime, reverse=True)

    if not saved_models:
        print(f"錯誤：在目錄 '{model_dir}' 中找不到任何 .pt 檔案。")
        exit()

    selected_model_path = questionary.select(
        "請選擇要評估的模型檔案 (預設為最新)：",
        choices=[str(p) for p in saved_models],
    ).ask()

    if selected_model_path is None:
        print("操作已取消。")
        exit()

    return selected_model_path, conv_layer_class, args


def evaluate(model_path: str, conv_layer_class: tuple[type[nn.Module], dict]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n將使用裝置: {device}")

    # 準備測試資料集
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 載入模型
    print(f"正在載入模型: {model_path}")
    model = ImageClassifierModel(conv_layer_class=conv_layer_class, num_classes=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="🧪 Testing", unit="batch", dynamic_ncols=True)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    end_time = time.time()

    # 計算平均損失和準確度
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    total_inference_time = end_time - start_time

    print("\n" + "=" * 40)
    print("📊 評估結果")
    print("=" * 40)
    print(f"模型檔案: {model_path}")
    print(f"測試集樣本數: {total_samples} 筆")
    print("-" * 40)
    print(f"平均損失 (Loss):    {avg_loss:.4f}")
    print(f"準確度 (Accuracy):   {accuracy:.2%}")
    print(f"總推論時間:          {total_inference_time:.2f} 秒")
    print("=" * 40)


if __name__ == "__main__":
    model_to_evaluate, selected_conv_class, args = get_user_model_selection()
    evaluate(model_to_evaluate, (selected_conv_class, args))
