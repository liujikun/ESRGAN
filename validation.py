import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn

import utils
import dataset

# ----------------------------------------
#                 Testing
# ----------------------------------------

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--wavelet_name', type = str, default = './models/dagan_1bi05_no_percep/Wavelet_iter40000_bs4.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--perceptualnet_name', type = str, default = './models/vgg16.pth', help = 'load the pre-trained model')
    parser.add_argument('--ESRGAN_name', type = str, default = './models/RRDB_ESRGAN_x4.pth', help = 'load the pre-trained model')
    # Initialization parameters
    parser.add_argument('--pad', type = str, default = 'zero', help = 'pad type of networks')
    parser.add_argument('--norm', type = str, default = 'none', help = 'normalization type of networks')
    parser.add_argument('--in_channels', type = int, default = 3, help = '1 for colorization, 3 for other tasks')
    parser.add_argument('--out_channels', type = int, default = 3, help = '2 for colorization, 3 for other tasks')
    parser.add_argument('--start_channels', type = int, default = 64, help = 'start channels for the main stream of generator')
    parser.add_argument('--wavelet_channels', type = int, default = 9, help = 'wavelet channels')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'initialization type of networks')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of networks')
    # Dataset parameters
    parser.add_argument('--baseroot_A', type = str, default = 'E:\\NTIRE 2020\\real-world super-resolution\\track 1\\Corrupted-tr-y', help = 'clean images set path')
    parser.add_argument('--baseroot_B', type = str, default = 'E:\\NTIRE 2020\\real-world super-resolution\\track 1\\Corrupted-tr-x', help = 'noisy images set path')
    parser.add_argument('--scale', type = int, default = 4, help = 'down sampling factor')
    parser.add_argument('--crop_size', type = int, default = 1024, help = 'crop patch size for HR patch')
    opt = parser.parse_args()

    # Define the dataset
    trainset = dataset.DomainTransferDataset(opt)
    print('The overall number of images:', len(trainset))

    # Define the basic variables
    img = trainset[0][0].unsqueeze(0).cuda()
    print(img.shape)
    net = utils.create_wavelet_generator(opt).cuda()
    out = net(img)
    out = out.squeeze(0).detach().permute(1,2,0).cpu().numpy() * 255
    out = out.astype(np.uint8)[:, :, [2, 1, 0]]
    cv2.imwrite("Image.png", out)
    