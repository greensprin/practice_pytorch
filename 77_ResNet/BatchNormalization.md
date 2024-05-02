[論文](https://arxiv.org/pdf/1502.03167)
[参考](https://cvml-expertguide.net/terms/dl/layers/batch-normalization-layer/)

# 概要
- ミニバッチ内のデータに対して、
    1. 各チャネル毎に特徴を正規化(標準化)
    2. スケール・シフトを行う

# 効果
- 学習の高速化
- 学習の安定化

NNは、データの分布が変わってしまうと学習をしなおさなくてはならなくなる

例えば、猫の画像を当てる場合、クロネコの画像だけで学習した場合、茶色などほかの色の猫も分類したい場合は学習しなおしになる

これを共変量シフトと呼ぶ

これは隠れ層でも同様のことが起きている。

前の層の重みが調整された場合、ある隠れ層に入力される値は変わってしまう

これにより再学習が必要となり、学習が安定せず、何度も調整が必要になるため遅くなる

これを改善するため、batch normalizationでは平均=0, 分散=1とすることにより少なくともその条件は守られたデータで学習することができるようになる。

これにより何もしないより分布が安定するため、学習も安定し、高速化される

# 処理順番
```mermaid
flowchart LR
    Conv --> BN --> ReLU
```

# 計算式
$$ data \space size = N $$

$$ mini \space batch = k = 1 ... n $$

mini batch 平均 (ch毎)

$$ \bar{x} = \sum_{i=1}^{n}x_i $$

mini batch 分散 (ch毎)

$$ \sigma_x = \frac{1}{n} \sum_{i=1}^{n}({x_i - \bar{x}})^2 $$

mini batch 標準化 (ch毎)

$$ \hat{x_i^{k}} = \frac{x_i^k - \bar{x}}{\sqrt{\sigma_x^2 + \epsilon}}$$

スケーリング・シフト

$$ y_i^k = \gamma^k\hat{x_i^k} + \beta^k$$

# 学習
スケーリングシフトで使う$\gamma$と$\beta$が学習されていく

# 推論
学習時のようにミニバッチでの処理を行えない(データ1つずつ処理するため)

推論ときは、データ全体から平均と分散を計算して正規化に使う

学習中にch毎にデータ全体の平均値、分散値を計算しておく

その後、以下の式で推論を行う

$$ E^c[x] \leftarrow E_\beta^c[\mu s]$$

$$ Var^c[x] \leftarrow \frac{m}{m-1} E_\beta^c[\sigma_\beta^c] $$

$$ y^c = \frac{\gamma^c}{\sqrt{Var^c[x]+\epsilon}} \cdot x^c + (\beta^c - \frac{\gamma E^c[x]}{\sqrt{Var^c[x] + \epsilon}}) $$

$\frac{m}{m-1}$は分散を不偏推定量にするための計算らしい

## 推論時の平均、分散の実際の計算方法
実際の計算では、移動平均が使われれているとのこと

すべてのデータの平均、分散を覚えておくのはメモリが非常に大きくなるため

移動平均であれば、直近の平均、分散と現在のミニバッチの平均、分散だけで計算できるため効率的。

最初mini batchの平均、分散はそのまま代入される

以降の平均、分散は以下のような感じで計算される

$$ ave_{next} = gamma * ave_{pre} + (1 - gamma) * ave_{now} $$

$$ var_{next} = gamma * var_{pre} + (1 - gamma) * var_{now} $$

$$ 0.0 \leqq gamma \leqq 1.0 \space (default = 0.99 or 0.9) $$

# 不偏推定量
参考:[不偏推定量](https://avilen.co.jp/personal/knowledge-article/unbiased-estimator/)

|用語|説明|
|:-:|:-|
|母数|母集団から求められる値。母集団の平均や分散などの値|
|不偏推定量|標本から測定した推定量の期待値が母数(母集団の平均や分散など)と等しくなるという性質をもった推定量|

## 平均の不偏推定量
標本を$x_1, x_2, ..., x_n$, 標本平均 = $\bar{x}$とすると

$$ E(\bar{x}) = E(\frac{1}{n}\sum_{i=1}^nx_i) = \frac{1}{n}\sum_{i=1}^nE(x_i) = \frac{1}{n} * n\mu = \mu $$

## 分散の不偏推定量
$$ E(s^2) = \frac{n-1}{n}\sigma^2$$