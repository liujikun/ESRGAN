#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares.

    * Affine deformation
    * Affine inverse deformation
    * Similarity deformation
    * Similarity inverse deformation
    * Rigid deformation
    * Rigid inverse deformation (* This algorithm is approximate, because the inverse formula 
                                   of the rigid deformation is not easy to infer)

For more details please reference the documentation: 
    
    Moving-Least-Squares/doc/Image Deformation.pdf

or the original paper: 
    
    Image deformation using moving least squares
    Schaefer, Mcphail, Warren. 

Note:
    In the original paper, the author missed the weight w_j in formular (5).
    In addition, all the formulars in section 2.1 miss the w_j. 
    And I have corrected this point in my documentation.

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

import numpy as np
from skimage.transform import rescale
import scipy.io
import glob

from PIL import Image
from PIL import ImageEnhance
from scipy.signal import convolve2d
from skimage.draw import circle
from scipy.ndimage import rotate


np.seterr(divide='ignore', invalid='ignore')
#import yaml
# from easydict import EasyDict as edict
# CONFIG = edict(yaml.load(open('blur_config.yaml', 'r')))
import random
import cv2
def mls_affine_deformation_1pt(p, q, v, alpha=1):
    ''' Calculate the affine deformation of one point.   
    This function is used to test the algorithm.
    '''
    ctrls = p.shape[0]
    np.seterr(divide='ignore')
    w = 1.0 / np.sum((p - v) ** 2, axis=1) ** alpha
    w[w == np.inf] = 2**31-1
    pstar = np.sum(p.T * w, axis=1) / np.sum(w)
    qstar = np.sum(q.T * w, axis=1) / np.sum(w)
    phat = p - pstar
    qhat = q - qstar
    reshaped_phat1 = phat.reshape(ctrls, 2, 1)
    reshaped_phat2 = phat.reshape(ctrls, 1, 2)
    reshaped_w = w.reshape(ctrls, 1, 1)
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)
    try:
        inv_pTwp = np.linalg.inv(pTwp)
    except np.linalg.linalg.LinAlgError:
        if np.linalg.det(pTwp) < 1e-8:
            new_v = v + qstar - pstar
            return new_v
        else:
            raise
    mul_left = v - pstar
    mul_right = np.sum(reshaped_phat1 * reshaped_w * qhat[:, np.newaxis, :], axis=0)
    new_v = np.dot(np.dot(mul_left, inv_pTwp), mul_right) + qstar
    return new_v


def mls_affine_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Affine deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Precompute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]

    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwp = np.sum(reshaped_phat1 * reshaped_w * reshaped_phat2, axis=0)                 # [2, 2, grow, gcol]
    try:                
        inv_pTwp = np.linalg.inv(pTwp.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False                
    except np.linalg.linalg.LinAlgError:                
        flag = True             
        det = np.linalg.det(pTwp.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf                
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwp[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwp = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    mul_left = reshaped_v - pstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = reshaped_w * reshaped_phat1                                             # [ctrls, 2, 1, grow, gcol]
    reshaped_mul_right =mul_right.transpose(0, 3, 4, 1, 2)                              # [ctrls, grow, gcol, 2, 1]
    A = np.matmul(np.matmul(reshaped_mul_left, inv_pTwp), reshaped_mul_right)           # [ctrls, grow, gcol, 1, 1]
    reshaped_A = A.reshape(ctrls, 1, grow, gcol)                                        # [ctrls, 1, grow, gcol]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = np.sum(reshaped_A * qhat, axis=0) + qstar                            # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0


    
    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    
    transformers = transformers*2/(float)(image.shape[0]-1) - 1
    
    return transformed_image, transformers

def mls_affine_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Affine inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]

    reshaped_phat = phat.reshape(ctrls, 2, 1, grow, gcol)                               # [ctrls, 2, 1, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    pTwq = np.sum(reshaped_phat * reshaped_w * reshaped_qhat, axis=0)                   # [2, 2, grow, gcol]
    try:
        inv_pTwq = np.linalg.inv(pTwq.transpose(2, 3, 0, 1))                            # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(pTwq.transpose(2, 3, 0, 1))                                 # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(1, 1, grow, gcol)                                    # [1, 1, grow, gcol]
        adjoint = pTwq[[[1, 0], [1, 0]], [[1, 1], [0, 0]], :, :]                        # [2, 2, grow, gcol]
        adjoint[[0, 1], [1, 0], :, :] = -adjoint[[0, 1], [1, 0], :, :]                  # [2, 2, grow, gcol]
        inv_pTwq = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                       # [grow, gcol, 2, 2]
    mul_left = reshaped_v - qstar                                                       # [2, grow, gcol]
    reshaped_mul_left = mul_left.reshape(1, 2, grow, gcol).transpose(2, 3, 0, 1)        # [grow, gcol, 1, 2]
    mul_right = np.sum(reshaped_phat * reshaped_w * reshaped_phat2, axis=0)             # [2, 2, grow, gcol]
    reshaped_mul_right =mul_right.transpose(2, 3, 0, 1)                                 # [grow, gcol, 2, 2]
    temp = np.matmul(np.matmul(reshaped_mul_left, inv_pTwq), reshaped_mul_right)        # [grow, gcol, 1, 2]
    reshaped_temp = temp.reshape(grow, gcol, 2).transpose(2, 0, 1)                      # [2, grow, gcol]

    # Get final image transfomer -- 3-D array
    transformers = reshaped_temp + pstar                                                # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image






def mls_similarity_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)        # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2), 
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

     # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    transformers = reshaped_temp / reshaped_mu  + qstar                                 # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]

    return transformed_image


def mls_similarity_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Similarity inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)                                                          # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
        adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
        adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

    v_minus_qstar_mul_mu = (reshaped_v - qstar) * reshaped_mu                           # [2, grow, gcol]
    
    # Get final image transfomer -- 3-D array
    reshaped_v_minus_qstar_mul_mu = v_minus_qstar_mul_mu.reshape(1, 2, grow, gcol)      # [1, 2, grow, gcol]
    transformers = np.matmul(reshaped_v_minus_qstar_mul_mu.transpose(2, 3, 0, 1),
                            inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1) + pstar    # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0

    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]

    # Rescale image
    transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')

    return transformed_image


def mls_rigid_deformation(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    sum_w = np.sum(w, axis=0)                                                           # [grow, gcol]
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / sum_w                # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat = phat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_left = np.concatenate((reshaped_phat, reshaped_neg_phat_verti), axis=1)         # [ctrls, 2, 2, grow, gcol]
    vpstar = reshaped_v - pstar                                                         # [2, grow, gcol]
    reshaped_vpstar = vpstar.reshape(2, 1, grow, gcol)                                  # [2, 1, grow, gcol]
    neg_vpstar_verti = vpstar[[1, 0],...]                                               # [2, grow, gcol]
    neg_vpstar_verti[1,...] = -neg_vpstar_verti[1,...]                                  
    reshaped_neg_vpstar_verti = neg_vpstar_verti.reshape(2, 1, grow, gcol)              # [2, 1, grow, gcol]
    mul_right = np.concatenate((reshaped_vpstar, reshaped_neg_vpstar_verti), axis=1)    # [2, 2, grow, gcol]
    reshaped_mul_right = mul_right.reshape(1, 2, 2, grow, gcol)                         # [1, 2, 2, grow, gcol]
    A = np.matmul((reshaped_w * mul_left).transpose(0, 3, 4, 1, 2), 
                       reshaped_mul_right.transpose(0, 3, 4, 1, 2))                     # [ctrls, grow, gcol, 2, 2]

    # Calculate q
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [2, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol).transpose(0, 3, 4, 1, 2)      # [ctrls, grow, gcol, 1, 2]

    # Get final image transfomer -- 3-D array
    temp = np.sum(np.matmul(reshaped_qhat, A), axis=0).transpose(2, 3, 0, 1)            # [1, 2, grow, gcol]
    reshaped_temp = temp.reshape(2, grow, gcol)                                         # [2, grow, gcol]
    norm_reshaped_temp = np.linalg.norm(reshaped_temp, axis=0, keepdims=True)           # [1, grow, gcol]
    norm_vpstar = np.linalg.norm(vpstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = reshaped_temp / norm_reshaped_temp * norm_vpstar  + qstar            # [2, grow, gcol]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0
    
    # Mapping original image
    transformed_image = np.ones_like(image) * 255
    new_gridY, new_gridX = np.meshgrid((np.arange(gcol) / density).astype(np.int16), 
                                        (np.arange(grow) / density).astype(np.int16))
    transformed_image[tuple(transformers.astype(np.int16))] = image[new_gridX, new_gridY]    # [grow, gcol]
    
    transformers = transformers*2/(float)(image.shape[0]-1) - 1
    print(type(transformers))
    # transformers = np.transpose(transformers, ())
    return transformed_image, transformers

def mls_rigid_deformation_inv(image, p, q, alpha=1.0, density=1.0):
    ''' Rigid inverse deformation
    ### Params:
        * image - ndarray: original image
        * p - ndarray: an array with size [n, 2], original control points
        * q - ndarray: an array with size [n, 2], final control points
        * alpha - float: parameter used by weights
        * density - float: density of the grids
    ### Return:
        A deformed image.
    '''
    height = image.shape[0]
    width = image.shape[1]
    # Change (x, y) to (row, col)
    q = q[:, [1, 0]]
    p = p[:, [1, 0]]

    # Make grids on the original image
    gridX = np.linspace(0, width, num=int(width*density), endpoint=False)
    gridY = np.linspace(0, height, num=int(height*density), endpoint=False)
    vy, vx = np.meshgrid(gridX, gridY)
    grow = vx.shape[0]  # grid rows
    gcol = vx.shape[1]  # grid cols
    ctrls = p.shape[0]  # control points

    # Compute
    reshaped_p = p.reshape(ctrls, 2, 1, 1)                                              # [ctrls, 2, 1, 1]
    reshaped_q = q.reshape((ctrls, 2, 1, 1))                                            # [ctrls, 2, 1, 1]
    reshaped_v = np.vstack((vx.reshape(1, grow, gcol), vy.reshape(1, grow, gcol)))      # [2, grow, gcol]
    
    w = 1.0 / np.sum((reshaped_p - reshaped_v) ** 2, axis=1)**alpha                     # [ctrls, grow, gcol]
    w[w == np.inf] = 2**31 - 1
    pstar = np.sum(w * reshaped_p.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    phat = reshaped_p - pstar                                                           # [ctrls, 2, grow, gcol]
    qstar = np.sum(w * reshaped_q.transpose(1, 0, 2, 3), axis=1) / np.sum(w, axis=0)    # [2, grow, gcol]
    qhat = reshaped_q - qstar                                                           # [ctrls, 2, grow, gcol]
    reshaped_phat1 = phat.reshape(ctrls, 1, 2, grow, gcol)                              # [ctrls, 1, 2, grow, gcol]
    reshaped_phat2 = phat.reshape(ctrls, 2, 1, grow, gcol)                              # [ctrls, 2, 1, grow, gcol]
    reshaped_qhat = qhat.reshape(ctrls, 1, 2, grow, gcol)                               # [ctrls, 1, 2, grow, gcol]
    reshaped_w = w.reshape(ctrls, 1, 1, grow, gcol)                                     # [ctrls, 1, 1, grow, gcol]

    mu = np.sum(np.matmul(reshaped_w.transpose(0, 3, 4, 1, 2) * 
                          reshaped_phat1.transpose(0, 3, 4, 1, 2), 
                          reshaped_phat2.transpose(0, 3, 4, 1, 2)), axis=0)             # [grow, gcol, 1, 1]
    reshaped_mu = mu.reshape(1, grow, gcol)                                             # [1, grow, gcol]
    neg_phat_verti = phat[:, [1, 0],...]                                                # [ctrls, 2, grow, gcol]
    neg_phat_verti[:, 1,...] = -neg_phat_verti[:, 1,...]                                
    reshaped_neg_phat_verti = neg_phat_verti.reshape(ctrls, 1, 2, grow, gcol)           # [ctrls, 1, 2, grow, gcol]
    mul_right = np.concatenate((reshaped_phat1, reshaped_neg_phat_verti), axis=1)       # [ctrls, 2, 2, grow, gcol]
    mul_left = reshaped_qhat * reshaped_w                                               # [ctrls, 1, 2, grow, gcol]
    Delta = np.sum(np.matmul(mul_left.transpose(0, 3, 4, 1, 2), 
                             mul_right.transpose(0, 3, 4, 1, 2)), 
                   axis=0).transpose(0, 1, 3, 2)                                        # [grow, gcol, 2, 1]
    Delta_verti = Delta[...,[1, 0],:]                                                   # [grow, gcol, 2, 1]
    Delta_verti[...,0,:] = -Delta_verti[...,0,:]
    B = np.concatenate((Delta, Delta_verti), axis=3)                                    # [grow, gcol, 2, 2]
    try:
        inv_B = np.linalg.inv(B)                                                        # [grow, gcol, 2, 2]
        flag = False
    except np.linalg.linalg.LinAlgError:
        flag = True
        det = np.linalg.det(B)                                                          # [grow, gcol]
        det[det < 1e-8] = np.inf
        reshaped_det = det.reshape(grow, gcol, 1, 1)                                    # [grow, gcol, 1, 1]
        adjoint = B[:,:,[[1, 0], [1, 0]], [[1, 1], [0, 0]]]                             # [grow, gcol, 2, 2]
        adjoint[:,:,[0, 1], [1, 0]] = -adjoint[:,:,[0, 1], [1, 0]]                      # [grow, gcol, 2, 2]
        inv_B = (adjoint / reshaped_det).transpose(2, 3, 0, 1)                          # [2, 2, grow, gcol]

    vqstar = reshaped_v - qstar                                                         # [2, grow, gcol]
    reshaped_vqstar = vqstar.reshape(1, 2, grow, gcol)                                  # [1, 2, grow, gcol]

    # Get final image transfomer -- 3-D array
    temp = np.matmul(reshaped_vqstar.transpose(2, 3, 0, 1),
                     inv_B).reshape(grow, gcol, 2).transpose(2, 0, 1)                   # [2, grow, gcol]
    norm_temp = np.linalg.norm(temp, axis=0, keepdims=True)                             # [1, grow, gcol]
    norm_vqstar = np.linalg.norm(vqstar, axis=0, keepdims=True)                         # [1, grow, gcol]
    transformers = temp / norm_temp * norm_vqstar + pstar                               # [2, grow, gcol]

    # Correct the points where pTwp is singular
    if flag:
        blidx = det == np.inf    # bool index
        transformers[0][blidx] = vx[blidx] + qstar[0][blidx] - pstar[0][blidx]
        transformers[1][blidx] = vy[blidx] + qstar[1][blidx] - pstar[1][blidx]

    # Removed the points outside the border
    transformers[transformers < 0] = 0
    transformers[0][transformers[0] > height - 1] = 0
    transformers[1][transformers[1] > width - 1] = 0
    # Mapping original image
    transformed_image = image[tuple(transformers.astype(np.int16))]    # [grow, gcol]
    # Rescale image
    # transformed_image = rescale(transformed_image, scale=1.0 / density, mode='reflect')
    transformers = transformers*2/(float)(image.shape[0]-1) - 1

    return transformed_image, transformers


def debug_point(landmark, img):
    import cv2
    for point in landmark:
        cv2.circle(img, (point[0], point[1]), 1, (255, 255, 255), 1)
    return img


'''
0:清晰　
1:模糊
2:噪声
3:模糊＋噪声
4:超分
5:模糊＋超分
6:噪声＋超分
7:模糊＋超分＋噪声
'''

def op4_DefocusBlur_random(img):
    kernelidx = np.random.randint(0, len(defocusKernelDims))
    kerneldim = defocusKernelDims[kernelidx]
    return DefocusBlur(img, kerneldim)


def DefocusBlur(img, dim):
    imgarray = np.array(img, dtype="float32")
    kernel = DiskKernel(dim)
    convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def DiskKernel(dim):
    kernelwidth = dim  #维度（2或者4或者6）
    kernel = np.zeros((kernelwidth, kernelwidth), dtype=np.float32)  #dim维的二维方阵
    circleCenterCoord = dim // 2  #取中心  1或者2或者3
    circleRadius = circleCenterCoord + 1  #变成0或者1或者2，半径

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)  #画圆，给定圆心与半径
    kernel[rr, cc] = 1  #圆的范围内全部置位1

    if (dim == 3 or dim == 5):
        kernel = Adjust(kernel, dim)

    normalizationFactor = np.count_nonzero(kernel)
    kernel = kernel / normalizationFactor  #又是全部加起来为1
    return kernel  


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel


class Distortion:

    def __init__(self, distortion):
        self.distortion_strength = distortion.strength
        self.label = [0, 0, 0]
        self.distortion = list()
        self.down = 1

    def op1_gaussian_blur(self, img,  bias=0):

        left = self.distortion_strength['Blur'][0]
        right = self.distortion_strength['Blur'][1]
        sigma = random.randint(left, right) + bias
        norm_sigma = float(sigma - left) / float(right - left)
        self.distortion.append(norm_sigma)
        if sigma >= 1 and sigma <= 6:
            sizeG = sigma * 12 + 1
            self.label[2] = 1
            return cv2.GaussianBlur(img, (sizeG, sizeG), sigma)
        return img

    def op2_down(self, img):
        left = self.distortion_strength.Down[0]
        right = self.distortion_strength.Down[1]
        op = random.randint(left, right)
        norm_op = float(op - left) / float(right - left)
        self.distortion.append(norm_op)

        scaleimg = 2**op
        self.down = scaleimg
        if scaleimg > 1:
            self.label[0] = 1
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    def op2_up(self, img, image_size):
        return cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)

    def op3_gaussian_noise(self, img):
        left = self.distortion_strength.Noise[0]
        right = self.distortion_strength.Noise[1]
        row, col, ch = img.shape
        mean = 0
        # var = 0.1
        # sigma = var ** 0.5
        var = random.randint(left, right)
        norm_var = float(var - left) / float(right - left)
        self.distortion.append(norm_var)
        if var == 0:
            return img
        sigma = var
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255
        self.label[1] = 1
        return noisy.astype(np.uint8)

    def op4_DefocusBlur_random(self, img):
        kernelidx = np.random.randint(0, len(defocusKernelDims))
        kerneldim = defocusKernelDims[kernelidx]
        return DefocusBlur(img, kerneldim)

    # define constant distortion
    def op1_gaussian_blur_without_rand(self, img):
        sigma = self.distortion_strength['Blur'][1]
        self.distortion.append(sigma/4.0)
        self.distortion.append(sigma)
        if sigma >= 1 and sigma <= 6:
            sizeG = sigma * 12 + 1
            self.label[2] = 1
            return cv2.GaussianBlur(img, (sizeG, sizeG), sigma)
        else:
            return img

    def op2_down_without_rand(self, img):
        op = self.distortion_strength['Down'][1]
        scaleimg = 2 ** op
        self.distortion.append(scaleimg/2.0)
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)

    def op3_gaussian_noise_without_rand(self, img):
        row, col, ch = img.shape
        mean = 0
        var = self.distortion_strength['Noise'][1]
        self.distortion.append(var/4.0)
        sigma = var
        # Fix time seed
        np.random.seed(5)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = img + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255
        self.label[1] = 1
        return noisy.astype(np.uint8)

    def Distort_random(self, image, Size):
        return self.op2_up(self.op3_gaussian_noise(
            self.op2_down(self.op1_gaussian_blur(image))), Size).astype(np.uint8), self.down

    def Distort_constant_noise(self, image, Size):
        return self.op2_up(self.op3_gaussian_noise_without_rand(self.op2_down_without_rand(image))
                           , Size).astype(np.uint8), self.down

        # constant
    def Distort_constant_blur(self, image, Size):
        return self.op2_up(self.op2_down_without_rand(self.op1_gaussian_blur_without_rand(image))
                           , Size).astype(np.uint8), self.down

    def Distort_constant_down(self, image, Size):
        return self.op2_up(self.op2_down_without_rand(image)
                           , Size).astype(np.uint8), self.down

    # constant
    def Distort_constant(self, image, Size):
        return self.op2_up(self.op3_gaussian_noise_without_rand(self.op2_down_without_rand(self.op1_gaussian_blur_without_rand(image)))
                           , Size).astype(np.uint8), self.down


class Distortion_v2:
    """
        New version of distortion
    """

    def __init__(self, distortion, ratio=1.):
       # self.distortion_strength = distortion.strength  #强度
        self.distortion_strength = distortion
        self.down = 1
        self.kernel_files = glob.glob("./utils/kernels/*.mat") 
        self.kernel_changes = list(np.linspace(1.0, 2.0, 5)) #生成5个从1到2的间隔数
        self.kernel_angles = list(np.linspace(0, 360, 9))  #生成9个从0到360的间隔数

        # low light enhance  #增加亮度
        #if distortion.low_light:
        if distortion.low_light:
            self.gamma_range = list(np.linspace(0.5, 0.9, 5))
            self.brightness_parameter = list(np.linspace(0.2, 1.0, 6))

        # scale ratio of degree
        self.ratio = ratio   #尺度比例
        #self.low_light = distortion.low_light
        self.low_light = distortion.low_light

    def _gama_trans(self, img, gama=2.2):
        img = np.power(img / 255., gama)  #先归一化到0-1,，然后求其2.2次方
        return img * 255.   #再乘回0-255的区间

    def Distort_random(self, img, img_size):
        self.o_size = img_size
        global_noise_flag = random.random()
        motion_blur_flag = random.random()
        defocus_flag = random.random()

        x = img
        if defocus_flag > 0.5:
            x = self._op_defocus(x)
        if motion_blur_flag > 0.5:
            x = self._op_motion(x)

        if global_noise_flag > 0.5:
            x = self._op_gaussian_noise(x)
        else:
            x = self._op_ychannel(x)

        # Downsize and upsize
        x, down_flag = self._op_down(x)

        # JEPG compression
        x = self._op_jpeg(x)

        if down_flag:
            x = self._op_up(x)

        return x, self.down

    def Distort_random_v2(self, img, img_size):
        self.o_size = img_size
        x = img
        # defocus blur
        x = self._op_defocus(x)
        # Downsize
        x, down_flag = self._op_down(x)
        # Noise
        x = self._op_gaussian_noise(x)

        if down_flag:
            x = self._op_up(x)

        return x, self.down

    def Distort_random_v3(self, img, img_size):
        self.o_size = img_size
        x = img
        # low light enhance   #亮度增强   这里是0，所以不处理
        if self.low_light:
            x = self.op4_darken(x)
        # defocus blur
        chance = random.randint(0, 5)   #随机初始化一个选择
        # 1/5 chance for motion blur
        if chance == 0:
            x = self._op_motion(x)
        # 1/5 chance for added
        elif chance == 1:
            # x = self._op_blur_motion(x)
            x = self._op_motion(x)  #增加运动模糊
            x = self._op_defocus(x)   #增加散焦
        # 1/5 chance
        elif chance == 2:    #此时1/5的几率不处理
            pass
        # 2/5 chance for defocus blur  
        else:    #2/5的几率只有defocus
            x = self._op_defocus(x)   #对应的参数Defocus只能是奇数开始，也就是[3,]

        # Downsize  下采样
       # x, down_flag = self._op_down(x)

        # Noise   #加噪声
        x = self._op_gaussian_noise(x)

        # JPEG compress   #jpg压缩
        x = self._op_jpeg(x)

        # beautify
        if self.distortion_strength.d_Beautify:
            x = self._op_beautyfy(x)

        #if down_flag:   #如果有下采样过
       #     x = self._op_up(x)   #就再上回来

        return x, self.down

    def _op_down(self, img):
        """Randomly downsample input image"""
#        left = self.distortion_strength.Down[0]

        left = self.distortion_strength.d_Down[0]
       # right = self.distortion_strength.Down[1]
        right = self.distortion_strength.d_Down[1]
        if right <= 0:
            return img, False
        op = random.randint(left, right)
        norm_op = float(op - left) / float(right - left)

        if op == 0:
            down_flag = False
        else:
            down_flag = True

        scaleimg = 2 ** op
        self.down = scaleimg
        # if scaleimg > 1:
        # self.label[0] = 1
        newh = img.shape[0] // scaleimg
        neww = img.shape[1] // scaleimg
        return cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA), down_flag

    def _op_up(self, img):
        """Resize up the image, if necessary. Work with `_op_down`"""
        return cv2.resize(img, (self.o_size[0], self.o_size[1]),
                          interpolation=cv2.INTER_LINEAR)

    def _op_gaussian_noise(self, img, strength=2):
        """The 3 channel color noise"""
#        left = self.distortion_strength.Noise[0]

        left = self.distortion_strength.d_Noise[0]
#        right = self.distortion_strength.Noise[1] * strength * self.ratio

        right = self.distortion_strength.d_Noise[1] * strength * self.ratio
        right = np.ceil(right)

        col, row, ch = img.shape
        mean = 0
        # scale = random.randint(2, 5)
        scale = random.randint(1, 5)
        var = random.randint(left, right)
        if var == 0:
            return img
        gauss = np.random.normal(mean, var, (int(row / scale), int(col / scale), ch))
        gauss = gauss.reshape(int(row / scale), int(col / scale), ch)
        gauss = cv2.resize(gauss, (row, col), interpolation=cv2.INTER_LINEAR)

        # Addition
        noisy = img + gauss
        noisy[noisy < 0] = 0
        noisy[noisy > 255] = 255

        return noisy.astype(np.uint8)

    def _op_defocus(self, img, TYPE='uniform', radius=15):
        """ Defocus _opteration"""
        left = self.distortion_strength.d_Defocus[0]  #2
#        left = self.distortion_strength.Defocus[0]

        right = np.ceil(self.distortion_strength.d_Defocus[1] * self.ratio)  #8
#        right = self.distortion_strength.Defocus[1] * self.ratio


        defocusKernelDims = [i for i in range(left, int(right), 2)]   #[2, 4, 6]
        TYPES = ['uniform', 'circle']   #失焦的类型
        TYPE = TYPES[np.random.randint(0, 2)]  #随机选一种
        #print(TYPE)
        # (3, 9)
        radius = np.random.randint(np.ceil(left/2), np.ceil(right/2))  #随机产生一个大小在0,2之间的半径（不包括2），这里只能是0或者1
        #print('radius:{}'.format(radius))
        sizeX, sizeY, channel_num = img.shape  #长 宽 深
        x, y = np.mgrid[-(radius + 1):(radius + 1), -(radius + 1):(radius + 1)]  #生成两个2(radius+1)大小的方阵，且互相转置
        # construct uniform disk kernel
        if TYPE == 'uniform':  #如果是uniform类型的散焦
            disk = (np.sqrt(x ** 2 + y ** 2) < radius).astype(float)  #在radius范围内的，都是1，其他都是0
            disk /= disk.sum()  #使得全部加起来才为1
        elif TYPE == 'circle':  #如果是circle类型的散焦
            # circle disk kernel
            kernelidx = np.random.randint(0, len(defocusKernelDims))  #随机调一种kernelDim
            #print('kernelidx:{}'.format(kernelidx))
            kerneldim = defocusKernelDims[kernelidx]  #获得挑出的kernelDim  2或者4或者6
            disk = DiskKernel(dim=kerneldim)
        else:
            raise NotImplementedError
        # gama transfer
        img = self._gama_trans(img, gama=2.2)
        smoothed = cv2.filter2D(img, -1, disk)   #使用kernel进行卷积滤波
        smoothed = self._gama_trans(smoothed, gama=1/2.2)
        return smoothed.astype(np.uint8)

    def _op_motion(self, img):   #送进来是一张图[512, 512, 3]
        """ Motion blur: all random variables generated inside the function"""
        left = self.distortion_strength.d_Motion[0]  #3
#        left = self.distortion_strength.Motion[0]

        right = np.ceil(self.distortion_strength.d_Motion[1] * self.ratio)   #5
#        right = np.ceil(self.distortion_strength.Motion[1] * self.ratio)

        # image = np.array(img)
        image = self._gama_trans(img, gama=2.2)  #[1440,1440,3]  还是0-255，但是值会偏小
        degree = random.randint(left, int(right))  #degree  3,4或者5  核的大小
        angle = random.randint(0, 360)   #0-360

        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)   #求得以degree/2为中心，以angle为旋转角度，不缩放的旋转矩阵

        motion_blur_kernel = np.diag(np.ones(degree))   #生成degree维的单位阵，如果degree=0的话，他就是个空
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))  #对单位矩阵进行变换，然后大小也是degree维矩阵

        # if motion_blur_kernel[0][0] == 0.0:
            # return img

        motion_blur_kernel = motion_blur_kernel / degree  #变换后的矩阵/degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel) #得到模糊后的矩阵，核就是motion_blur_kernel
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)  #对图像进行正则化
        blurred = np.array(blurred)  #转为了numpy数组
        # gama
        img = self._gama_trans(blurred, gama=1/2.2)  #开头_gama_trans的反步骤
        return img.astype(np.uint8)

    def _op_blur_motion(self, img):
        img = self._gama_trans(img, 2.2)
        # load kernel
        blur_kernel = scipy.io.loadmat(random.choice(self.kernel_files))["kernel"]
        w, h = blur_kernel.shape

        ratio_ = random.choice(self.kernel_changes) * self.ratio
        new_shape = (int(w * ratio_), int(h * ratio_))
        blur_kernel = cv2.resize(blur_kernel, new_shape, interpolation=cv2.INTER_LINEAR)

        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=0)
        if random.random() > 0.5:
            blur_kernel = np.flip(blur_kernel, axis=1)
        if random.random() > 0.5:
            blur_kernel = rotate(blur_kernel, random.choice(self.kernel_angles))

        blur_kernel /= blur_kernel.sum()
        img = cv2.filter2D(img, -1, blur_kernel)
        img = self._gama_trans(img, 1/2.2)
        return img.astype(np.uint8)

    def _op_ychannel(self, img, strength=2):

        image_ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
        left = self.distortion_strength['Noise'][0]
#        left = self.distortion_strength.Noise[0]

        right = self.distortion_strength['Noise'][1] * strength
#        right = self.distortion_strength.Noise[1] * strength

        row, col, ch = image_ycbcr.shape

        mean = 0
        var = random.randint(left, right)
        sigma = var
        scale = random.randint(1, 3)
        norm_var = float(var - left) / float(right - left)
        if var == 0:
            return img

        gauss = np.random.normal(mean, sigma, (int(row / scale), int(col / scale), 1))
        gauss = gauss.reshape(int(row / scale), int(col / scale), 1)
        gauss = cv2.resize(gauss, (row, col), interpolation=cv2.INTER_LINEAR)

        noisy_y = image_ycbcr[:, :, 0] + gauss
        noisy_y = np.clip(noisy_y, 0, 255)
        image_ycbcr[:, :, 0] = noisy_y
        noisy = image_ycbcr

        noisy = cv2.cvtColor(noisy, cv2.COLOR_YCR_CB2RGB)

        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)

    def _op_jpeg(self, img):
        # 4/9 without jpeg compression
        jpegq = random.randint(0, 50)
        if jpegq < 25 or jpegq > 45:
            return img
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpegq]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encimg, 1)

    def op4_darken(self, img):
        """
            Darken the image
        """

        # Directly use PIL Image tools to adjust the brightness
        img_array = img.astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        brightness_enhance = ImageEnhance.Brightness(pil_img)
        b_param = random.choice(self.brightness_parameter)  # brightness change
        pil_img = brightness_enhance.enhance(b_param)
        img_array = np.asarray(pil_img)

        # Gamma transformation
        gamma_ = random.choice(self.gamma_range)
        img_array = ((img_array.astype(np.float32) / 255.0) ** (1 / gamma_)) * 255.0
        random.randrange()
        return img_array.astype(np.uint8)

    def _op_beautyfy(self, img):
        # beatuify enhance:  r = 2-7,  eps=0.1^2-0.4^2, s=1, 2
        # 4/9 without jpeg compression
        chance = random.randint(0, 1)
        # 0.05-0.4
        eps = random.randint(self.distortion_strength.d_Beautify[0], self.distortion_strength.d_Beautify[1])*0.1
        r = random.randint(2, 7)
        s = random.randint(1, 2)
        if chance == 0:
            return img
        img = img / 255.0
        return (cv2.ximgproc.guidedFilter(img.astype(np.float32),img.astype(np.float32), r, eps**2, s)*255.0).astype(np.uint8)


if __name__ == '__main__':
    image = cv2.imread('/data/wangchao4/face_hd_part1/1_女_15-30岁_不戴眼镜/正脸/IMG_2244_crop_285.png')
    # img = op4_DefocusBlur_random(image)
    class strngth:
        def __init__(self):
            self.strength = 1
    strn = strngth()
    dis2 = Distortion_v2(strn)
    img = dis2._op_defocus(image, TYPE='circle', radius=15)
    cv2.imwrite('disk_blur_cr.png', img)

