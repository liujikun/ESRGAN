import os
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import cv2
import network

def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content



def create_ESRGAN_generator(opt):
    # Initialize the network
    esrgan = network.RRDBNet(3, 3, 64, 23, gc = 32)
    # Init the network
    if opt.ESRGAN_name:
        pretrained_net = torch.load(opt.ESRGAN_name)
        load_dict(esrgan, pretrained_net)
    else:
        network.weights_init(esrgan, init_type = opt.init_type, init_gain = opt.init_gain)
    print('ESRGAN generator is created!')
    return esrgan

def create_ESRGAN_discriminator(opt):
    # Initialize the network
    discriminator = network.SR_VGG128_Discriminator(in_nc = 3,base_nf = 64)
    # Init the network
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('ESRGAN discriminator is created!')
    return discriminator

def create_perceptualnet(opt):
    # Get the first 23 layers of vgg16, which is conv4_3
    perceptualnet = network.MINCFeatureExtractor()
    # Pre-trained VGG-16
    # if opt.perceptualnet_name:
    #     vgg16_dict = torch.load(opt.perceptualnet_name)
    #     load_dict(perceptualnet, vgg16_dict)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    print('Perceptual network is created!')
    return perceptualnet

def load_dict(process_net, pretrained_dict):
    # Get the dict from pre-trained network
    #pretrained_dict = pretrained_dict
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net

def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1).cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)[0, :, :, :]
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_img_path, img_copy)

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    # Load name
    parser.add_argument('--load_name', type = str, default = '', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--perceptualnet_name', type = str, default = './models/vgg16.pth', help = 'load the pre-trained model')
    parser.add_argument('--ESRGAN_name', type = str, default = './models/RRDB_ESRGAN_x4.pth', help = 'load the pre-trained model')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--wavelet_channels', type = int, default = 9, help = 'wavelet channels')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    opt = parser.parse_args()

    print('Next:')
    net = create_wavelet_generator(opt).cuda()
    a = torch.randn(1, 3, 64, 64).cuda()
    b = net(a)
    print(b.shape)
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)
    
    print('Next:')
    net = create_wavelet_discriminator(opt).cuda()
    a = torch.randn(1, 9, 32, 32).cuda()
    b = net(a)
    print(b.shape)
    a = torch.randn(1, 9, 128, 128).cuda()
    b = net(a)
    print(b.shape)

    print('Next:')
    net = create_ESRGAN_generator(opt).cuda()
    a = torch.randn(1, 3, 64, 64).cuda()
    b = net(a)
    print(b.shape)

    print('Next:')
    net = create_ESRGAN_discriminator(opt).cuda()
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)

    print('Next:')
    net = create_perceptualnet(opt).cuda()
    a = torch.randn(1, 3, 64, 64).cuda()
    b = net(a)
    print(b.shape)
    a = torch.randn(1, 3, 256, 256).cuda()
    b = net(a)
    print(b.shape)
