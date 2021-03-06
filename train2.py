import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--mode', type = int, default = 1, help = '3 modes: domain adaptation GAN / SR / jointly training')
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--task_name', type = str, default = 'dagan', help = 'task name for loading networks, saving, and log')
    parser.add_argument('--save_mode', type = str, default = 'iter', help = 'saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type = int, default = 5, help = 'interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type = int, default = 10000, help = 'interval between model checkpoints (by iterations)')
    parser.add_argument('--wavelet_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--perceptualnet_name', type = str, default = './models/vgg16.pth', help = 'load the pre-trained model')
    parser.add_argument('--ESRGAN_name', type = str, default = './models/RRDB_ESRGAN_x4.pth', help = 'load the pre-trained model')
    # GPU parameters
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type = str, default = '0, 1, 2, 3', help = 'gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 1000, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 0.0002, help = 'Adam: learning rate for G')
    parser.add_argument('--lr_d', type = float, default = 0.0002, help = 'Adam: learning rate for D')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: decay of second order momentum of gradient')
    parser.add_argument('--lr_decrease_mode', type = str, default = 'iter', help = 'lr decrease mode, by_epoch or by_iter')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_iter', type = int, default = 50000, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor')
    parser.add_argument('--num_workers', type = int, default = 4, help = 'number of cpu threads to use during batch generation')
    # Loss parameters
    parser.add_argument('--lambda_wavelet_l1', type = float, default = 1, help = 'coefficient for Wavelet low-pass L1 Loss')
    parser.add_argument('--lambda_wavelet_gan', type = float, default = 0.01, help = 'coefficient for high-pass GAN Loss')
    parser.add_argument('--lambda_wavelet_percep', type = float, default = 0.005, help = 'coefficient for Wavelet low-pass perceptual Loss')
    parser.add_argument('--lambda_sr_l1', type = float, default = 0.01, help = 'coefficient for SR L1 Loss')
    parser.add_argument('--lambda_sr_gan', type = float, default = 0.005, help = 'coefficient for SR GAN Loss')
    parser.add_argument('--lambda_sr_percep', type = float, default = 1, help = 'coefficient for SR perceptual Loss')
    # GAN parameters
    parser.add_argument('--gan_mode', type = str, default = 'LSGAN', help = 'type of GAN: [LSGAN | WGAN], LSGAN is recommended')
    parser.add_argument('--additional_training_d', type = int, default = 1, help = 'number of training D more times than G')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--wavelet_channels', type = int, default = 1, help = 'wavelet channels')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_A', type = str, default = '../track 1/Corrupted-tr-y', help = 'clean images set path')
    parser.add_argument('--baseroot_B', type = str, default = '../track 1/Corrupted-tr-x', help = 'noisy images set path')
    parser.add_argument('--scale', type = int, default = 4, help = 'down sampling factor')
    parser.add_argument('--crop_size', type = int, default = 1024, help = 'crop patch size for HR patch')
    opt = parser.parse_args()

    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    
    # ----------------------------------------
    #                 CycleGAN
    # ----------------------------------------
    if opt.mode == 1:
        trainer.WaveletGAN_kernel_Trainer(opt)
