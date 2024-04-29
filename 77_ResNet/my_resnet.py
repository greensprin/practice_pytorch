'''
テンプレートの使い方
1. モデルを作成する. Sequentialの中に処理の流れを記載する
2. データを準備する. DataLoaderを作成する
3. 誤差関数(default=CrossEntropyLoss)、最適化関数(SGD)で何を使うかを考える
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F
from collections import OrderedDict
from torchinfo import summary

import os
from tqdm import tqdm

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=ch),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        fx  = self.model(x)
        hx  = fx + x
        out = self.relu(hx)
        return out

class DownSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2), # down sampling layer
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_ch),
        )

        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=2), # down sampling layer for input
            nn.BatchNorm2d(num_features=out_ch),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x_down_sampling = self.downsampling(x)
        fx = self.model(x)
        hx = fx + x_down_sampling
        out = self.relu(hx)
        return out

class MyNet(nn.Module):
    def __init__(self, layer_num=3):
        super().__init__()

        od = OrderedDict()

        # =======================================
        # 1. First Layer (1 Conv layer)
        # =======================================
        # Conv (kernel=3x3 ch=16 output=32x32)
        od["first"]      = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        od["BN_first"]   = nn.BatchNorm2d(num_features=16)
        od["ReLU_first"] = nn.ReLU()

        # =======================================
        # 2. Second Layer (2n Conv layers)
        # =======================================
        # Conv (kernel=3x3 ch=16 output=32x32) * 2n
        for i in range(layer_num):
            od[f"first_block{i}"] = ResBlock(16)

        # =======================================
        # 3. Second Layer (1 subsumpling Conv layer + (2n-1) Conv layers)
        # =======================================
        # Conv (kernel=1x1 ch=32 output=16x16 stride=2) subsmpling layer
        od[f"downsampling1"] = DownSamplingBlock(16, 32)

        # Conv (kernel=3x3 ch=32 output=16x16) * (2n - 2)
        for i in range(layer_num - 1):
            od[f"second_block{i}"] = ResBlock(32)

        # =======================================
        # 4. Second Layer (1 subsumpling Conv layer + (2n-1) Conv layers)
        # =======================================
        # Conv (kernel=1x1 ch=64 output=8x8 stride=2) subsmpling layer
        od[f"downsampling2"] = DownSamplingBlock(32, 64)

        # Conv (kernel=3x3 ch=64 output=8x8) * (2n - 2)
        for i in range(layer_num - 1):
            od[f"third_block{i}"] = ResBlock(64)

        self.model = nn.Sequential(od)

        # =======================================
        # 5. Fully Connection Layer
        # =======================================
        # AveragePooling
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # Flatten
        self.flatten = nn.Flatten()

        # LinearLayer
        self.fc = nn.Linear(in_features=64, out_features=10)

        # SoftMax
        self.softmax = nn.Softmax(dim=1)

        # Note: This model has 6n+2 Layers.

    def forward(self, x):
        x = self.model(x)
        x = self.adaptive_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x

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
            current = batch + 1
            iter_num =  round(size / len(X)) if (batch == 0) else iter_num
            print(f"loss: {loss:>7f} [{current:>5d}] / {iter_num:>5d}]")

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

def downLoadCifar10():
    train_preprocess = transforms.Compose([
        transforms.RandomCrop(size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616), (0.2470322324632823, 0.24348512800005553, 0.2615878417279641)),
    ])

    test_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616), (0.2470322324632823, 0.24348512800005553, 0.2615878417279641)),
    ])

    train_data = datasets.CIFAR10(
        root      = "../data/cifar10",
        train     = True,
        download  = True,
        transform = train_preprocess,
    ) # 50000 (3[ch] * 32[width] * 32[height])

    test_data  = datasets.CIFAR10(
        root      = "../data/cifar10",
        train     = False,
        download  = True,
        transform = test_preprocess,
    ) # 10000 (3[ch] * 32[width] * 32[height])

    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers=2)
    test_dataloader  = DataLoader(test_data , batch_size = batch_size, shuffle = True, num_workers=2)

    return train_dataloader, test_dataloader

def main():
    # =================================
    # 1. 使用するデバイスを指定
    # =================================
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # =================================
    # 2. データ取得
    # =================================
    train_dataloader, test_dataloader = downLoadCifar10()

    # =================================
    # 3. モデル生成
    # =================================
    model = MyNet(layer_num=3).to(device)
    summary(model=model, input_size=(128, 3, 32, 32), depth=5)

    # =================================
    # 4. モデルのロード
    # =================================
    model_save_path = "model/model.pth" # モデルの保存先
    if   (os.path.exists(model_save_path) == True):
        model.load_state_dict(torch.load(model_save_path))
    elif (os.path.exists(os.path.dirname(model_save_path)) == False):
        os.makedirs(os.path.dirname(model_save_path))

    # =================================
    # 5. 誤差関数
    # =================================
    loss_fn = nn.CrossEntropyLoss()

    # =================================
    # 6. 最適化関数
    # =================================
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay=0.0001)

    # 32K(epoch=91), 48K(epoch=136)でそれぞれlrを1/10する必要がある。
    # 64K(epoch=182)で学習は終わり
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[91, 136], gamma=0.1)

    # =================================
    # 7. 学習
    # =================================
    epochs = 182
    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test (device, test_dataloader , model, loss_fn)
        scheduler.step()
    print("Done!")

    # =================================
    # 8. モデルの保存
    # =================================
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved Pytorch Model State to {model_save_path}")

if __name__ == "__main__":
    main()