# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 10:10
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model.model import net_G, net_D
from utils import params
from utils import *
import time

def predicate(args, train_data, train_cond, save_path):
    train_gt, test_gt, train_condtion, test_condtion = train_test_split(train_data, train_cond, test_size=0.1, random_state=42)

    def array2tensor(data_temp):
        data_temp = torch.from_numpy(data_temp).float()
        return data_temp

    test_gt = array2tensor(test_gt)
    test_condtion = array2tensor(test_condtion)

    test_dataset = TensorDataset(test_gt, test_condtion)

    G = net_G(args)
    G.to(params.device)
    G.load_state_dict(torch.load('/data1/zzd/simple-pytorch-3dgan-master/outputs/dcgan/first_test/models/G.pth', map_location='cuda:1'))
    loss_G = nn.L1Loss()
    G.eval()

    i = 0
    total_loss = 0
    time_list = []
    for (X, X_cond) in test_dataset:
        Z = generateZ(args, 1)
        Z = Z.view(-1, 400, 1, 1, 1)
        X = X.view(-1, 1, 64, 64, 64).to(params.device)
        X_cond_inp = X_cond.view([X.size(0), 3, 1, 1, 1]).to(params.device)
        G_input_fake = torch.cat([Z, X_cond_inp], dim=1).to(params.device)
        start_time = time.time()
        fake = G(G_input_fake)
        end_time = time.time()
        if i > 0:
            time_list.append(end_time - start_time)
        loss = loss_G(fake, X)
        total_loss += loss
        slice_index = 32

        if i % 10 == 0:
            voxels = fake.cpu().detach().squeeze().numpy()
            # 创建一个包含3个子图的图表
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 绘制XY平面的切片
            axes[0].imshow(voxels[:, :, slice_index], cmap='hot')
            axes[0].set_title(f'XY Plane at Slice {slice_index}')

            # 绘制XZ平面的切片
            axes[1].imshow(voxels[:, slice_index, :], cmap='hot', aspect='auto')
            axes[1].set_title(f'XZ Plane at Slice {slice_index}')

            # 绘制YZ平面的切片
            axes[2].imshow(voxels[slice_index, :, :], cmap='hot', aspect='auto')
            axes[2].set_title(f'YZ Plane at Slice {slice_index}')

            plt.savefig(save_path + '/{}.png'.format(str(i).zfill(3)))
            plt.close()
        i += 1

    print(f"The average loss is {total_loss / len(test_dataset)}")
    print(f"The average time is {sum(time_list) / len(time_list)}")






