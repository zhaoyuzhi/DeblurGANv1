import time
import datetime
import os
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

    # configurations
    save_folder = os.path.join(opt.save_path, opt.task_name)
    sample_folder = os.path.join(opt.sample_path, opt.task_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

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
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'DeblurGANv1_pre_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size)
        if opt.save_mode == 'iter':
            model_name = 'DeblurGANv1_pre_iter%d_bs%d.pth' % (iteration, opt.train_batch_size)
        save_model_path = os.path.join(opt.save_path, opt.task_name, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    #opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.DeblurDataset(opt, 'train')
    valset = dataset.DeblurDataset(opt, 'val')
    print('The overall number of training images:', len(trainset))
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(train_loader):

            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input)
            
            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)

            # Overall Loss and optimize
            loss = Pixellevel_L1_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))

def Continue_train_WGAN(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = os.path.join(opt.save_path, opt.task_name)
    sample_folder = os.path.join(opt.sample_path, opt.task_name)
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

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
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), 'DeblurGANv1_wgan_epoch%d_bs%d.pth' % (epoch, opt.batch_size))
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), 'DeblurGANv1_wgan_iter%d_bs%d.pth' % (iteration, opt.train_batch_size))
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), 'DeblurGANv1_wgan_epoch%d_bs%d.pth' % (epoch, opt.train_batch_size))
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), 'DeblurGANv1_wgan_iter%d_bs%d.pth' % (iteration, opt.train_batch_size))
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    #opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.DeblurDataset(opt, 'train')
    valset = dataset.DeblurDataset(opt, 'val')
    print('The overall number of training images:', len(trainset))
    print('The overall number of validation images:', len(valset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size = opt.train_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    val_loader = DataLoader(valset, batch_size = opt.val_batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target) in enumerate(train_loader):
            
            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Train Discriminator
            for j in range(opt.additional_training_d):

                optimizer_D.zero_grad()

                # Generator output
                fake_target = generator(true_input)

                # Fake samples
                fake_scalar_d = discriminator(true_input, fake_target.detach())
                true_scalar_d = discriminator(true_input, true_target)
                # Overall Loss and optimize
                loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
                loss_D.backward()

            # Train Generator
            optimizer_G.zero_grad()
            fake_target = generator(true_input)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target)
            
            # GAN Loss
            fake_scalar = discriminator(true_input, fake_target)
            GAN_Loss = - torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss = opt.lambda_l1 * Pixellevel_L1_Loss + GAN_Loss
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [GAN Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), GAN_Loss.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), optimizer_D)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'train_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [fake_target, true_target]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
