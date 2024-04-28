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

import os

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
        )

    def forward(self, x):
        x = self.model(x)
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
    # =================================
    # 1. 使用するデバイスを指定
    # =================================
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # =================================
    # 2. データ取得
    # =================================
    # TODO

    # =================================
    # 3. モデル生成
    # =================================
    model = MyNet().to(device)

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
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    # =================================
    # 7. 学習
    # =================================
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test (device, test_dataloader , model, loss_fn)
    print("Done!")

    # =================================
    # 8. モデルの保存
    # =================================
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved Pytorch Model State to {model_save_path}")

if __name__ == "__main__":
    main()