```mermaid
mindmap
    root(AI)
        基本
            パーセプトロン
            NN
            CNN
            DNN
        応用
            auto encoder
            attension
        モデル
            画像処理
                GAN
                VAE
                diffusion
                vision transformer
            自然言語
                RNN
                seq2seq
                transformer
```

# モデルを組む流れ

```mermaid
flowchart TB
    A[仕様デバイス指定]
    B[データ作成]
    C[DataLoader作成]
    D[モデル定義]
    E[誤差関数定義]
    F[最適化関数定義]
    G[モデル学習]
    H[モデル保存]

    A --> B --> 
    C --> D -->
    E --> F -->
    G --> H
```

# モデル学習の流れ

```mermaid
flowchart TB
    A[データをデバイスに送る]
    B[モデル推論-forward]
    C[誤差計算]
    D[誤差逆伝播-backward]
    E[パラメータ調整]
    F[勾配初期化]

    A --> B --> C --> D --> E --> F
```

# モデル作成

以下のような感じで作成する

```python
class MyNet(nn.Module):
    if __init__(self):
        super().__init__() # 親クラスの初期化 (必須らしい)

        self.model = nn.Sequential(
            # ここに組みたい処理を入れる (以下は例)
            nn.Linear(20, 30), # in_feature, out_feature
            nn.ReLU(),         # out_featureの形そのまま(引数不要)
            nn.Linear(30, 10), # in_feature(前段のout_feature), out_feature
            nn.ReLU(),         # 同上
            nn.Linear(10, 5),  # 同上
        )

        def forward(self, x):
            # nn.Sequentialを使うと、forwardの処理は以下のように書くだけでよい
            # 必要であればモデルの初段に合わせてnn.Flatten()などでデータの変形を行う
            pred = self.model(x)
            return pred

        # backwardは定義する必要はない
        # 最後の処理の結果に対してbackwardを実施してあげることで、
        # その地点からさかのぼって自動でgradを計算してくれるらしい
```