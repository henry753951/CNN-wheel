import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from python.CUDA.base_cconv2d import Cconv2d
from models.baseline_model import Net
import time

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 CIFAR-10 數據集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 定義自定義模型
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = Cconv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = Cconv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 訓練和測試函數
def train_and_test(model, device, trainloader, testloader, epochs=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 訓練
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}')

    # 測試
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')
    return accuracy

# 計時函數
def measure_time(model, data):
    model.eval()
    inputs = data[0].to(device)
    start = time.time()
    with torch.no_grad():
        _ = model(inputs)
    end = time.time()
    return end - start

# 訓練和測試內建 Conv2d 模型
net = Net()
print("Training Built-in Conv2d Model:")
accuracy_builtin = train_and_test(net, device, trainloader, testloader)

# 訓練和測試自定義 Conv2d 模型
custom_net = CustomNet()
print("\nTraining Custom Conv2d Model:")
accuracy_custom = train_and_test(custom_net, device, trainloader, testloader)

# 計時比較
data = next(iter(testloader))
time_builtin = measure_time(net, data)
time_custom = measure_time(custom_net, data)
print(f'\nBuilt-in Conv2d Time: {time_builtin:.4f}s')
print(f'Custom Conv2d Time: {time_custom:.4f}s')
print(f'Built-in Conv2d Accuracy: {accuracy_builtin}%')
print(f'Custom Conv2d Accuracy: {accuracy_custom}%')