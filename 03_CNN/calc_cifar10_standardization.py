import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

import os
import numpy as np
import cv2

def downLoadCifar10():
    preprocess = transforms.Compose([
        transforms.ToTensor(), # This functions converts PIL image or numpy arrya in range [0, 255] to Torch.Float.Tensor in range [0.0, 1.0]

        # # Calc from Train and Test Data
        # transforms.Normalize((0.49186877885008395, 0.48265390516493006, 0.44717727749693653),
        #                      (0.2469712143255279 , 0.24338893940434994, 0.2615925905215076 )),

        # Calc from Train Data Only
        transforms.Normalize((0.49139967861519745, 0.4821584083946076, 0.44653091444546616),
                             (0.2470322324632823, 0.24348512800005553, 0.2615878417279641)),

        # # Calc from Test Data Only
        # transforms.Normalize((0.4942142800245097, 0.48513138901654346, 0.4504090927542889),
        #                      (0.24665251509497976, 0.24289226346005363, 0.2615923780220238)),
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

    batch_size = 16
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers=2)
    test_dataloader  = DataLoader(test_data , batch_size = batch_size, shuffle = True, num_workers=2)

    return train_data, test_data, train_dataloader, test_dataloader

def calc_std(data):
    mean = data.mean()
    std  = np.sqrt(np.sum((data - mean) ** 2) / (data.size))
    return std, mean
    
if __name__ == "__main__":
    train_data, test_data, train_dataloader, test_dataloader = downLoadCifar10()

    # preprocessで定義した処理は、dataloaderから読みだされるたびに処理される
    # そのため、r_data_trainなどを直接見た場合は、値が0 ~ 255の範囲になっている
    # for X, y in train_dataloader:
    #     print(X[0])
    #     quit()

    r_data_train = train_data.data[:, :, :, 0] / 255
    g_data_train = train_data.data[:, :, :, 1] / 255
    b_data_train = train_data.data[:, :, :, 2] / 255

    r_data_test = test_data.data[:, :, :, 0] / 255
    g_data_test = test_data.data[:, :, :, 1] / 255
    b_data_test = test_data.data[:, :, :, 2] / 255
    
    r_data = np.zeros((60000, 32, 32))
    g_data = np.zeros((60000, 32, 32))
    b_data = np.zeros((60000, 32, 32))

    r_data[:50000, :, :] = r_data_train
    g_data[:50000, :, :] = g_data_train
    b_data[:50000, :, :] = b_data_train

    r_data[50000:, :, :] = r_data_test
    g_data[50000:, :, :] = g_data_test
    b_data[50000:, :, :] = b_data_test

    print("=============================")
    print("Calc from Train and Test Data")
    print("=============================")
    std_r, mean_r = calc_std(r_data)
    std_g, mean_g = calc_std(g_data)
    std_b, mean_b = calc_std(b_data)

    print(f"(({mean_r}, {mean_g}, {mean_b}), ({std_r}, {std_g}, {std_b}))")

    print("=============================")
    print("Calc from Train Data Only")
    print("=============================")
    std_r, mean_r = calc_std(r_data_train)
    std_g, mean_g = calc_std(g_data_train)
    std_b, mean_b = calc_std(b_data_train)

    print(f"(({mean_r}, {mean_g}, {mean_b}), ({std_r}, {std_g}, {std_b}))")

    print("=============================")
    print("Calc from Test Data Only")
    print("=============================")
    std_r, mean_r = calc_std(r_data_test)
    std_g, mean_g = calc_std(g_data_test)
    std_b, mean_b = calc_std(b_data_test)

    print(f"(({mean_r}, {mean_g}, {mean_b}), ({std_r}, {std_g}, {std_b}))")

    # 画像表示
    # zoom = 20
    # cv2.imshow("img", cv2.cvtColor(train_data.data[0].repeat(zoom, axis=0).repeat(zoom, axis=1), cv2.COLOR_RGB2BGR))
    # cv2.waitKey()