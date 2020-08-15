import os
import time
import datetime
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_wavelets import DWTForward
from tensorboardX import SummaryWriter
import cv2
import dataset
import utils
def add_noise(opt,x):
    # TODO: support more method of adding noise.
    if self.flag_add_noise==False:
        return x

    if hasattr(self, '_d_'):
        self._d_ = self._d_ * 0.9 + torch.mean(self.fx_tilde).item() * 0.1
    else:
        self._d_ = 0.0
    strength = 0.2 * max(0, self._d_ - 0.5)**2
    z = np.random.randn(*x.size()).astype(np.float32) * strength
    z = Variable(torch.from_numpy(z)).cuda() if self.use_cuda else Variable(torch.from_numpy(z))
    return x + z
def ESRGAN_Trainer(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = os.path.join(opt.save_path, opt.task_name)
    sample_folder = os.path.join(opt.sample_path, opt.task_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()

    # Initialize networks
    generator = utils.create_ESRGAN_generator(opt)
    discriminator = utils.create_ESRGAN_discriminator(opt)
    perceptualnet = utils.create_perceptualnet(opt)


    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()

    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        # Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'Wavelet_epoch%d_bs%d.pth' % (epoch, opt.batch_size)
        if opt.save_mode == 'iter':
            model_name = 'Wavelet_iter%d_bs%d.pth' % (iteration, opt.batch_size)
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

    # Define the dataset
    trainset = dataset.LRHRDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Count start time
    prev_time = time.time()

    # Tensorboard
    writer = SummaryWriter()

    # For loop training
    for epoch in range(opt.epochs):        

        # Record learning rate
        for param_group in optimizer_G.param_groups:
            writer.add_scalar('data/lr', param_group['lr'], epoch)
            print('learning rate = ', param_group['lr'])
        
        if epoch == 0:
            iters_done = 0

        for i, (img_LR, img_HR) in enumerate(dataloader):

            # To device
            # A is for downsample clean image, B is for noisy image
            #assert img_LR.shape == img_HR.shape
            img_LR = img_LR.cuda()
            img_HR = img_HR.cuda()

            # Adversarial ground truth
            valid = Tensor(np.ones((img_LR.shape[0], 1, img_LR.shape[2] // 8, img_LR.shape[3] // 8)))
            fake = Tensor(np.zeros((img_LR.shape[0], 1, img_LR.shape[2] // 8, img_LR.shape[3] // 8)))
            z = np.random.randn(opt.batch_size,1,128,128).astype(np.float32)
            z = np.repeat(z, 64, axis=1)
            # z = np.repeat(z, 32, axis=3)
            z = Variable(torch.from_numpy(z)).cuda() 
            # print(z.size())
            ### Train Generator
            # Forward
            pred = generator(img_LR,z)

            # L1 loss
            loss_L1 = criterion_L1(pred, img_HR)

            # gan part
            fake_scalar = discriminator(pred)
            loss_gan = criterion_MSE(fake_scalar, valid)

            # Perceptual loss part
            fea_true = perceptualnet(img_HR)
            fea_pred = perceptualnet(pred)
            # print(fea_pred.size())
            loss_percep = criterion_MSE(fea_true, fea_pred)

            # Overall Loss and optimize
            optimizer_G.zero_grad()
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_gan * loss_gan + opt.lambda_percep * loss_percep
            loss.backward()
            optimizer_G.step()

            ### Train Discriminator
            # Forward
            pred = generator(img_LR,z)
            
            # GAN loss
            fake_scalar = discriminator(pred.detach())
            loss_fake = criterion_MSE(fake_scalar, fake)
            true_scalar = discriminator(img_HR)
            loss_true = criterion_MSE(true_scalar, valid)
            
            # Overall Loss and optimize
            optimizer_D.zero_grad()
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()
            optimizer_D.step()
            
            # Record losses
            writer.add_scalar('data/loss_L1', loss_L1.item(), iters_done)
            writer.add_scalar('data/loss_percep', loss_percep.item(), iters_done)
            writer.add_scalar('data/loss_G', loss.item(), iters_done)
            writer.add_scalar('data/loss_D', loss_D.item(), iters_done)

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + i
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [G Loss: %.4f] [G percep Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                ((epoch + 1), opt.epochs, i, len(dataloader), loss_L1.item(), loss_gan.item(), loss_percep.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + 1), (iters_done + 1), optimizer_D)

            ### Sample data every epoch
            if (epoch + 1) % 1 == 0: 
                img_list = [pred, img_HR]
                name_list = ['pred', 'gt']
                utils.save_sample_png(sample_folder = sample_folder, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

    writer.close()
