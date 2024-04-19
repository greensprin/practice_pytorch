# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
# opencv
import cv2
# other
import os
import numpy as np

# global variable
debug = 0

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # データ平坦化関数
        # N(batch_size)以外の次元を1つ次元に平坦化する
        # torch.flatten()もある。こちらはすべての次元を平坦化するというところがnn.Flattenと異なる
        # nn.Flattenをそのまま使う場合、nn.Flatten()(x)のようになるので、__init__でself.flattenなどに代入しておいた方が直観的ではあるっぽい
        # torch.flatten()は、torch.flatten(x)のように使えるので、そのまま使ったほうが良い?
        self.flatten = nn.Flatten()

        # モデル定義
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.model(x)
        return logits

def train(device, dataloader, model, loss_fn, optimizer):
    # データのすべてのサイズ (60000)
    size = len(dataloader.dataset)

    # modelをトレーニングモードにする (gradient tracking on)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # データを指定のデバイスに配置 (modelと同じデバイスに配置しないとエラーになる)
        X, y = X.to(device), y.to(device)

        # 推論(modelの処理) (forward)
        pred = model(X) # nn.Module内の__call__でforwardが実行されることが定義されているため、この書き方でforwardが実行されると思われる

        # モデルの推論値の誤差を計算
        loss = loss_fn(pred, y)

        # 学習(パラメータ調整) (backward)
        loss.backward()       # 計算した誤差から、各パラメータに対する勾配を計算する
        optimizer.step()      # 計算した勾配を基にパラメータ(weight, bias)を調整する
        optimizer.zero_grad() # 各tensorの勾配(grad)を0にする

        '''
        backward()を実行すると、その地点から入力までの各所の偏微分を実施する
        これにより、誤差(loss)を小さくする勾配の方向、大きさが計算される
        その後、optimizer.step()により、計算した勾配を基にパラメータが調整される
        もし、pred.backward()とした場合は、誤差計算の前の状態から偏微分の計算が行われる (あまり意味ないが、backwardを実行した地点からさかのぼって偏微分するということを言いたい)
        step()でパラメータを調整した後は、zero_grad()で計算した勾配を0にすることで、次の学習に備える
        pytorchは逆伝播を行う際、勾配を累積する。これは、RNNの訓練やミニバッチ処理の時に有効なためとのこと。ただし、今回はRNNではないので1回のミニバッチの処理の後にzero_gradで勾配を初期化してあげる
        '''

        # 進捗表示
        if (batch % 100 == 0):
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

def show_image(image, zoom = 1):
    image_ext = image.repeat(zoom, axis = 0).repeat(zoom, axis = 1)
    cv2.imshow("image", image_ext)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 使用するデバイスを指定
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # データダウンロード (FashionMNIST)
    train_data = datasets.FashionMNIST(
        root      = "data",
        train     = True,
        download  = True,
        transform = ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root      = "data",
        train     = False,
        download  = True,
        transform = ToTensor(),
    )

    if (debug == 1):
        print(f"train_data size: {len(train_data)}") # 60000
        print(f"test_data  size: {len(test_data )}") # 10000

    # DataLoader作成
    batch_size = 64
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True) # shuffle = Trueでデータをシャッフルできる
    test_dataloader  = DataLoader(test_data , batch_size = batch_size, shuffle = True)

    sample_image = np.zeros((28, 28))
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}") # N = batch size(64), C = ch size(1), H = height(28), W = width(28)
        print(f"Shape of y: {y.shape} {y.dtype}")    # 64, torch.int64
        sample_image = X[0, 0, :, :].numpy()         # FashionMNISTのバッチの最初の画像を取得 (後で画像表示する用)
        break

    # 画像表示 (FashionMNISTのデータを可視化する)
    if (debug == 1):
        show_image(sample_image, 20)

    # モデルインスタンス (to(device)で指定のデバイスにモデルデータを配置する)
    model = MyNet().to(device)

    # モデルのロード
    model_save_path = "model/model.pth" # モデルの保存先
    if (os.path.exists(model_save_path) == True):
        model.load_state_dict(torch.load(model_save_path))

    # 誤差関数
    loss_fn = nn.CrossEntropyLoss()

    # 最適化関数
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

    # モデルの学習
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------")
        train(device, train_dataloader, model, loss_fn, optimizer)
        test (device, test_dataloader , model, loss_fn)
    print("Done!")

    # モデルの保存
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved Pytorch Model State to {model_save_path}")

    # 結果の可視化
    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()

    test_input  = test_data[0][0]
    test_output = test_data[0][1]

    with torch.no_grad():
        test_input = test_input.to(device)

        pred = model(test_input)

        predicted = classes[pred[0].argmax(0)] # 最大数のインデックスを返す
        actual    = classes[test_output]

        print(f"Predicted: {predicted}") # 推論結果
        print(f"actual   : {actual}")    # 正解

    show_image(test_input[0].numpy(), 20) # 画像表示

if __name__ == "__main__":
    main()