# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 10:18
'''
main.py

Welcome, this is the entrance to 3dgan
'''

import argparse
from train import trainer
import torch
import numpy as np
from test import predicate
from utils import params

#这段代码需要取消注释
# train_label = np.load('/data1/zzd/3D_condition_diffusion_2/data/train_data.npy')#(1170,64,64,64)
# train_cond = np.load('/data1/zzd/3D_condition_diffusion_2/data/train_cond.npy')#(1170,3)
# train_data = train_label[1:1170,...]
# train_cond = train_cond[1:1170,...]
# save_path = '/data1/zzd/simple-pytorch-3dgan-master/outputs/predicate_results'

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # add arguments
    parser = argparse.ArgumentParser()

    # loggings parameters
    parser.add_argument('--logs', type=str, default='first_test', help='logs by tensorboardX')
    parser.add_argument('--local_test', type=str2bool, default=False, help='local test verbose')
    parser.add_argument('--model_name', type=str, default="dcgan_zhonghe", help='model name for saving')
    parser.add_argument('--test', type=str2bool, default=False, help='call tester.py')
    parser.add_argument('--use_visdom', type=str2bool, default=False, help='visualization by visdom')
    args = parser.parse_args()

    # list params
    params.print_params()

    # run program
    if not args.test:
        trainer(args)
    else:
        pass
        #predicate(args,train_data,train_cond,save_path)


if __name__ == '__main__':
    main()
