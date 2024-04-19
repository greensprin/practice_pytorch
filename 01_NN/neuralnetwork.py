import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() # 親クラスであるnn.Moduleの初期化

        self.model = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Softmax(dim=1),
        )       

    def forward(self, x):
        x = self.model(x)
        return x

def main():
    input  = torch.randn(128, 20)
    model  = NeuralNetwork()
    output = model(input)
    print(output.shape)

if __name__ == "__main__":
    main()