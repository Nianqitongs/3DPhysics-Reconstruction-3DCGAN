# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 10:10
'''
trainer.py

Train 3dgan models
'''

import torch
from torch import optim
from torch import nn
from utils import *
import os
from model.model import net_G, net_D
# added
import datetime
import time
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from utils import params
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from utils.utils import generateZ


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def trainer(args):
    # added for output dir
    save_file_path = params.output_dir + '/' + args.model_name
    print(save_file_path)  # ../outputs/dcgan
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)

    # for using tensorboard
    if args.logs:
        model_uid = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        # writer = SummaryWriter(params.output_dir+'/'+args.model_name+'/'+model_uid+'_'+args.logs+'/logs')
        writer = SummaryWriter(params.output_dir + '/' + args.model_name + '/' + args.logs + '/logs')

        image_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/images'
        model_saved_path = params.output_dir + '/' + args.model_name + '/' + args.logs + '/models'

        if not os.path.exists(image_saved_path):
            os.makedirs(image_saved_path)
        if not os.path.exists(model_saved_path):
            os.makedirs(model_saved_path)

    # datset define
    # dsets_path = args.input_dir + args.data_dir + "train/"
    dsets_path = params.data_dir + params.model_dir + "30/train/"
    # if params.cube_len == 64:
    #     dsets_path = params.data_dir + params.model_dir + "30/train64/"

    print(dsets_path)  # ../volumetric_data/chair/30/train/

    # --------------------------------------------------------
    train_label = np.load('/data1/zzd/3D_condition_diffusion_2/data/train_data.npy')#(1170,64,64,64)
    train_cond = np.load('/data1/zzd/3D_condition_diffusion_2/data/train_cond.npy')#(1170,3)
    train_data = train_label[1:1170,...]
    train_cond = train_cond[1:1170,...]

    train_gt,test_gt,train_condtion,test_condtion = train_test_split(train_data,train_cond,test_size=0.1,random_state=42)


    def array2tensor(data_temp):
        data_temp = torch.from_numpy(data_temp).float()
        return data_temp

    train_gt = array2tensor(train_gt)
    train_condtion = array2tensor(train_condtion)

    train_dataset = TensorDataset(train_gt, train_condtion)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=params.batch_size)  # 128
    # -------------------------------------------------------
    dset_len = {"train": len(train_dataset)}
    dset_loaders = {"train": gen}
    # print (dset_len["train"])

    # model define
    D = net_D(args)
    G = net_G(args)

    # print total number of parameters in a model
    # x = sum(p.numel() for p in G.parameters() if p.requires_grad)
    # print (x)
    # x = sum(p.numel() for p in D.parameters() if p.requires_grad)
    # print (x)

    D_solver = optim.Adam(D.parameters(), lr=params.d_lr, betas=params.beta)
    # D_solver = optim.SGD(D.parameters(), lr=args.d_lr, momentum=0.9)
    G_solver = optim.Adam(G.parameters(), lr=params.g_lr, betas=params.beta)

    D.to(params.device)
    G.to(params.device)

    # criterion_D = nn.BCELoss()
    criterion_D = nn.MSELoss()

    criterion_G = nn.L1Loss()

    itr_val = -1
    itr_train = -1

    for epoch in range(params.epochs):

        start = time.time()

        for phase in ['train']:
            if phase == 'train':
                # if args.lrsh:
                #     D_scheduler.step()
                D.train()
                G.train()
            else:
                D.eval()
                G.eval()

            running_loss_G = 0.0
            running_loss_D = 0.0
            running_loss_adv_G = 0.0

            for i, (X, X_cond) in enumerate(tqdm(dset_loaders[phase])):

                # if phase == 'val':
                #     itr_val += 1

                if phase == 'train':
                    itr_train += 1

                X = X.to(params.device)
                # print (X)
                # print (X.size())

                batch = X.size()[0]
                # print (batch)

                Z = generateZ(args, batch)
                Z = Z.view(-1, 400, 1, 1, 1)
                # print (Z.size())

                # ============= Train the discriminator =============#
                X = X.view(-1, 1, 64, 64, 64)
                X_cond_inp = X_cond.view([X.size(0), 3, 1, 1, 1]).to(params.device)
                D_inp_real = torch.cat([X, torch.ones_like(X) * X_cond_inp], dim=1).to(params.device)
                d_real = D(D_inp_real)

                G_input_fake = torch.cat([Z, X_cond_inp], dim=1).to(params.device)
                fake = G(G_input_fake)
                D_inp_fake = torch.cat([fake, torch.ones_like(fake) * X_cond_inp], dim=1).to(params.device)
                d_fake = D(D_inp_fake)

                real_labels = torch.ones_like(d_real).to(params.device)
                fake_labels = torch.zeros_like(d_fake).to(params.device)
                # print (d_fake.size(), fake_labels.size())

                if params.soft_label:
                    real_labels = torch.Tensor(batch).uniform_(0.7, 1.2).to(params.device)
                    fake_labels = torch.Tensor(batch).uniform_(0, 0.3).to(params.device)

                # print (d_real.size(), real_labels.size())
                d_real_loss = criterion_D(d_real, real_labels)

                d_fake_loss = criterion_D(d_fake, fake_labels)

                d_loss = d_real_loss + d_fake_loss

                # no deleted
                d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
                d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
                d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

                if d_total_acu < params.d_thresh:
                    D_solver.zero_grad()
                    d_loss.backward(retain_graph=True)
                    D_solver.step()

                # =============== Train the generator ===============#

                # Z = generateZ(args, batch)
                # Z = Z.view(-1,200,1,1,1)
                # G_input_fake = torch.cat([Z, X_cond_inp], dim=1).to(params.device)

                # # print (X)
                # fake = G(G_input_fake)  # generated fake: 0-1, X: 0/1
                input_fake_G = torch.cat([fake, torch.ones_like(fake) * X_cond_inp], dim=1).to(params.device)
                d_fake = D(input_fake_G)

                adv_g_loss = criterion_D(d_fake, real_labels)
                # print (fake.size(), X.size())

                # recon_g_loss = criterion_D(fake, X)
                recon_g_loss = criterion_G(fake, X)
                # g_loss = recon_g_loss + params.adv_weight * adv_g_loss
                g_loss = adv_g_loss + recon_g_loss * 10

                if args.local_test:
                    # print('Iteration-{} , D(x) : {:.4} , G(x) : {:.4} , D(G(x)) : {:.4}'.format(itr_train, d_loss.item(), recon_g_loss.item(), adv_g_loss.item()))
                    print('Iteration-{} , D(x) : {:.4}, D(G(x)) : {:.4}'.format(itr_train, d_loss.item(),
                                                                                adv_g_loss.item()))

                D_solver.zero_grad()
                G_solver.zero_grad()
                g_loss.backward(retain_graph=False)
                G_solver.step()

                # =============== logging each 10 iterations ===============#

                running_loss_G += recon_g_loss.item() * X.size(0)
                running_loss_D += d_loss.item() * X.size(0)
                running_loss_adv_G += adv_g_loss.item() * X.size(0)

                if args.logs:
                    loss_G = {
                        'adv_loss_G': adv_g_loss,
                        'recon_loss_G': recon_g_loss,
                    }

                    loss_D = {
                        'adv_real_loss_D': d_real_loss,
                        'adv_fake_loss_D': d_fake_loss,
                    }

                    # if itr_val % 10 == 0 and phase == 'val':
                    #     save_val_log(writer, loss_D, loss_G, itr_val)

                    if itr_train % 10 == 0 and phase == 'train':
                        save_train_log(writer, loss_D, loss_G, itr_train)

            # =============== each epoch save model or save image ===============#
            epoch_loss_G = running_loss_G / dset_len[phase]
            epoch_loss_D = running_loss_D / dset_len[phase]
            epoch_loss_adv_G = running_loss_adv_G / dset_len[phase]

            end = time.time()
            epoch_time = end - start

            print('Epochs-{} ({}) , D(x) : {:.4}, D(G(x)) : {:.4}'.format(epoch, phase, epoch_loss_D, epoch_loss_adv_G))
            print('Elapsed Time: {:.4} min'.format(epoch_time / 60.0))

            if (epoch + 1) % params.model_save_step == 0:
                print('model_saved, images_saved...')
                torch.save(G.state_dict(), model_saved_path + '/G.pth')
                torch.save(D.state_dict(), model_saved_path + '/D.pth')

                samples = fake.cpu().data[:1].squeeze().numpy()
                samples_real = X.cpu().data[:1].squeeze().numpy()
                # print (samples.shape)
                # image_saved_path = '../images'

                SavePloat_Voxels(samples, samples_real, image_saved_path, epoch)
