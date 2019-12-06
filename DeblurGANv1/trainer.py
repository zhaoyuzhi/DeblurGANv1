import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        remain_epoch = opt.epoch - opt.lr_decrease_epoch
        remain_epoch = opt.epoch - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr * remain_epoch / remain_epoch
            lr = max(lr, 1e-7)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'Pre_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'Pre_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DeblurDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(dataloader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            residual = generator(true_input)
            fake_target = true_input + residual
            
            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

def Continue_train_WGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        remain_epoch = opt.epoch - opt.lr_decrease_epoch
        remain_epoch = opt.epoch - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr * remain_epoch / remain_epoch
            lr = max(lr, 1e-7)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator.module, 'WGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator.module, 'WGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    if opt.save_name_mode:
                        torch.save(generator, 'WGAN_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                        print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    if opt.save_name_mode:
                        torch.save(generator, 'WGAN_iter%d_bs%d.pth' % (iteration, opt.batch_size))
                        print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.DeblurDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(dataloader):
            
            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Train Discriminator
            for j in range(opt.additional_training_d):
                optimizer_D.zero_grad()

                # Generator output
                residual = generator(true_input)
                fake_target = true_input + residual

                # Fake samples
                fake_scalar_d = discriminator(true_input, fake_target.detach())
                true_scalar_d = discriminator(true_input, true_target)
                # Overall Loss and optimize
                loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
                loss_D.backward()

            # Train Generator
            optimizer_G.zero_grad()
            residual = generator(true_input)
            fake_target = true_input + residual

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            
            # GAN Loss
            fake_scalar = discriminator(true_input, fake_target)
            GAN_Loss = - torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss + opt.lambda_gan * GAN_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [GAN Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), Pixellevel_L1_Loss.item(), GAN_Loss.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), optimizer_D)
