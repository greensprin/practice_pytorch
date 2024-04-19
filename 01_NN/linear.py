import torch
from torch import nn

'''

nn.Linear(in_feature, out_feature, bias=True, device=None, dtype=None)

in_feature : input  data size
out_feature: output data size

- in_featureに設定した数のデータを入れると、out_featureの数のデータとして出力される
- weightは(out_feature, in_feature)の行列になっている

'''
nn.Linear
model = nn.Linear(20, 30)

# 正規分布乱数でtensor行列を作成する
input = torch.randn(128, 20)
print(input)

# output = input * weight + bias
output = model(input)

print(output) # (128, 30)