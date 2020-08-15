import cv2
import numpy as np

def solve(src, kernel):
    dst = cv2.filter2D(src, -1, kernel)
    return dst

if __name__ == "__main__":

    # define path
    path = "E:\\NTIRE 2020\\real-world super-resolution\\track 1\\Corrupted-tr-x\\Flickr2K_000058.png"
    src = cv2.imread(path)
    
    # define kernels
    smooth_kernel1 = np.array((
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625]), dtype = np.float32)
    smooth_kernel2 = np.array((
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]), dtype = np.float32)
    sharpen_kernel1 = np.array((
        [-1/9, -1/9, -1/9],
        [-1/9, 17/9, -1/9],
        [-1/9, -1/9, -1/9]), dtype = np.float32)
    sharpen_kernel2 = np.array((
        [-1/3, -1/3, -1/3],
        [-1/3, 11/3, -1/3],
        [-1/3, -1/3, -1/3]), dtype = np.float32)
    sharpen_kernel3 = np.array((
        [-1/9, -1/9, -1/9, -1/9, -1/9],
        [-1/9, -1/9, -1/9, -1/9, -1/9],
        [-1/9, -1/9, 33/9, -1/9, -1/9],
        [-1/9, -1/9, -1/9, -1/9, -1/9],
        [-1/9, -1/9, -1/9, -1/9, -1/9]), dtype = np.float32)

    # func
    dst = solve(src, sharpen_kernel3)
    cv2.imwrite("dst.png", dst)
    '''
    cv2.imshow('dst_img', dst)
    cv2.waitKey(0)
    '''
