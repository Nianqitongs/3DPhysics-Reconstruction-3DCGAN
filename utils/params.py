# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 10:15
'''
params.py

Managers of all hyper-parameters

'''

import torch

epochs = 12000
batch_size = 1#64
soft_label = False
adv_weight = 0
d_thresh = 0.8
z_dim = 400#200
z_dis = "norm"
model_save_step = 20#1
g_lr = 0.00005
d_lr = 0.001
beta = (0.5, 0.999)
cube_len = 32#32
leak_value = 0.2
bias = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data_dir = '/data1/zzd/simple-pytorch-3dgan-master/volumetric_data/'
model_dir = 'chair/'  # change it to train on other data models
output_dir = '/data1/zzd/simple-pytorch-3dgan-master/outputs'

def print_params():
    l = 16
    print(l * '*' + 'hyper-parameters' + l * '*')
    print('epochs =', epochs)
    print('batch_size =', batch_size)
    print('soft_labels =', soft_label)
    print('adv_weight =', adv_weight)
    print('d_thresh =', d_thresh)
    print('z_dim =', z_dim)
    print('z_dis =', z_dis)
    print('model_images_save_step =', model_save_step)
    print('data =', model_dir)
    print('device =', device)
    print('g_lr =', g_lr)
    print('d_lr =', d_lr)
    print('cube_len =', cube_len)
    print('leak_value =', leak_value)
    print('bias =', bias)
    print(l * '*' + 'hyper-parameters' + l * '*')
