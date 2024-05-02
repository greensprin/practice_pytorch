import numpy as np
import matplotlib.pyplot as plot

def main():
    # 元入力
    input = np.random.default_rng().uniform(0, 255, (2, 100))

    # 入力 - 平均値
    input_sub_mean = input - input.mean()

    # 標準化後
    input_sub_mean_div_std = input_sub_mean / input.std()

    plot.scatter(input                 [0, :], input                 [1, :])
    plot.scatter(input_sub_mean        [0, :], input_sub_mean        [1, :])
    plot.scatter(input_sub_mean_div_std[0, :], input_sub_mean_div_std[1, :])
    plot.show()

if __name__ == "__main__":
    main()