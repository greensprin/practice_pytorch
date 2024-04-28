import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F

import os

class MyCNN(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.flatten = nn.Flatten()

        self.conv_layer = nn.Sequential(
            # Convolution Layer
            # layer 1 (Conv -> ReLU -> LRN -> MaxPooling)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),        # (in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0), # (size, alpha=0.0001, beta=0.75, k=1.0)
            nn.MaxPool2d(kernel_size=3, stride=2),                        # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

            # layer 2 (Conv -> ReLU -> LRN -> MaxPooling)
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),       # (in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0), # (size, alpha=0.0001, beta=0.75, k=1.0)

            # kernel_size = 3 -> 2
            nn.MaxPool2d(kernel_size=3, stride=2),                        # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

            # layer 3 (Conv -> ReLU)
            nn.Conv2d(256, 384, kernel_size=5, stride=1, padding=2),      # (in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ReLU(),

            # layer 4 (Conv -> ReLU)
            nn.Conv2d(384, 384, kernel_size=5, stride=1, padding=2),      # (in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ReLU(),

            # layer 5 (Conv -> ReLU -> MaxPooling)
            nn.Conv2d(384, 256, kernel_size=5, stride=1, padding=2),      # (in_channels, out_channels, kernel_size, stride=1, padding=0, ...)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),                        # (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        )

        self.fully_connected_layer = nn.Sequential(
            # Fully Connected Layer (全結合層)
            # layer 6
            nn.Dropout(p=0.5), # p = 0.5 (dropout 50%)
            nn.Linear(256 * 6 * 6, 4096), # in_feature = 256[ch] * 6[height] * 6[width]
            nn.ReLU(),

            # layer 7
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),

            # layer 8
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.flatten(x) # [batch_size, 256, 6, 6] -> [batch_size, 256 * 6 * 6]
        x = self.fully_connected_layer(x) # 全結合層に入る前にデータを1次元に平坦化してから入力する
        return x

def downLoadCifar10():
    preprocess = transforms.Compose([
        transforms.Resize(256),     # 32x32 -> 256x256
        transforms.CenterCrop(224), # 256x256 -> 224x224(center)
        transforms.ToTensor(),
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616), (0.2470322324632823, 0.24348512800005553, 0.2615878417279641)),
    ])

    train_data = datasets.CIFAR10(
        root      = "../data/cifar10",
        train     = True,
        download  = True,
        transform = preprocess,
    ) # 50000 (3[ch] * 32[width] * 32[height])

    test_data  = datasets.CIFAR10(
        root      = "../data/cifar10",
        train     = False,
        download  = True,
        transform = preprocess,
    ) # 10000 (3[ch] * 32[width] * 32[height])

    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers=2)
    test_dataloader  = DataLoader(test_data , batch_size = batch_size, shuffle = True, num_workers=2)

    return train_dataloader, test_dataloader

def train(device, dataloader, model, loss_fn, optimizer):
    # データのすべてのサイズ (50000)
    size = len(dataloader.dataset)

    # modelをトレーニングモードにする (gradient tracking on)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # データを指定のデバイスに配置 (modelと同じデバイスに配置しないとエラーになる)
        X, y = X.to(device), y.to(device)

        # 推論(modelの処理) (forward)
        pred = model(X)

        # モデルの推論値の誤差を計算
        loss = loss_fn(pred, y)

        # 学習(パラメータ調整) (backward)
        loss.backward()       # 計算した誤差から、各パラメータに対する勾配を計算する
        optimizer.step()      # 計算した勾配を基にパラメータ(weight, bias)を調整する
        optimizer.zero_grad() # 各tensorの勾配(grad)を0にする

        # 進捗表示
        if (batch % 10 == 0):
            loss = loss.item() # .item()は1つのtensorを値に変換するためのメソッド
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}] / {size:>5d}]")

def test(device, dataloader, model, loss_fn):
    # データのすべてのサイズ (10000)
    size = len(dataloader.dataset)

    num_batches = len(dataloader)
    
    # 評価モードに変更 (gradient tracking off)
    model.eval()

    test_loss = 0
    correct   = 0

    with torch.no_grad(): # 勾配を保持しない状態で実行する (メモリ消費を削減できる)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)

            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct   /= size

    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    # 使用するデバイスを指定
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # データ取得 (cifar10)
    train_dataloader, test_dataloader = downLoadCifar10()

    # モデル生成
    model = MyCNN(10).to(device)

    # モデルのロード
    model_save_path = "model/model.pth" # モデルの保存先
    if (os.path.exists(model_save_path) == True):
        model.load_state_dict(torch.load(model_save_path))
    elif (os.path.exists(os.path.dirname(model_save_path)) == False):
        os.makedirs(os.path.dirname(model_save_path))

    # 誤差関数
    loss_fn = nn.CrossEntropyLoss()

    # 最適化関数
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    # 学習
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test (device, test_dataloader , model, loss_fn)
    print("Done!")

    # モデルの保存
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved Pytorch Model State to {model_save_path}")

if __name__ == "__main__":
    main()