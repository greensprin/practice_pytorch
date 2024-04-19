import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as F

import os
import numpy as np
import cv2

def downLoadCifar10():
    train_data = datasets.CIFAR10(
        root      = "data/cifar10",
        train     = True,
        download  = True,
        transform = ToTensor(),
    ) # 50000 (3[ch] * 32[width] * 32[height])

    test_data  = datasets.CIFAR10(
        root      = "data/cifar10",
        train     = False,
        download  = True,
        transform = ToTensor(),
    ) # 10000 (3[ch] * 32[width] * 32[height])

    return train_data, test_data

def calc_std(data):
    mean = data.mean()
    std  = np.sqrt(np.sum((data - mean) ** 2) / (data.size))
    return std, mean
    
if __name__ == "__main__":
    train_data, test_data = downLoadCifar10()

    r_data = train_data.data[:, :, :, 0] / 255
    g_data = train_data.data[:, :, :, 1] / 255
    b_data = train_data.data[:, :, :, 2] / 255

    std_r, mean_r = calc_std(r_data)
    std_g, mean_g = calc_std(g_data)
    std_b, mean_b = calc_std(b_data)

    print(mean_r)
    print(mean_g)
    print(mean_b)

    print(std_r)
    print(std_g)
    print(std_b)

    # 画像表示
    # zoom = 20
    # cv2.imshow("img", cv2.cvtColor(train_data.data[0].repeat(zoom, axis=0).repeat(zoom, axis=1), cv2.COLOR_RGB2BGR))
    # cv2.waitKey()