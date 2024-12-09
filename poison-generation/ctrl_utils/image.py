import math
#from skimage import transform, data
import numpy as np
import cv2
#import tensorflow as tf
#from tensorflow.keras.applications import InceptionV3
#from skimage.measure import compare_ssim
#import bm3d
import scipy.signal
import torch
from PIL import Image


def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        #img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YUV)
        x_yuv[i] = img
    return x_yuv


def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        #img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YUV2RGB)
        x_rgb[i] = img
    return x_rgb






def DCT(x_train, window_size):
    # x_train: (idx, w, h, ch)

    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct

    return x_dct            # x_dct: (idx, ch, w, h)


def IDCT(x_train, window_size):
    # x_train: (idx, ch, w, h)
    x_idct = np.zeros(x_train.shape, dtype=np.float)

    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float))
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct




def grid_sample(x_train, grid):
    x_train = torch.tensor(x_train)
    x_train = x_train.permute(0, 3, 1, 2)
    x_train = torch.nn.functional.grid_sample(x_train, grid, align_corners=True)
    x_train = x_train.permute(0, 2, 3, 1).numpy()

    return x_train





def poison(x_train, y_train, param):
    print("Wrong Label Attack")
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])
    x_origin = x_train.copy()
    index = np.where(y_train != target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index], y_train[index], param)
    y_train[index] = target_label
    return x_train, index

def poison_clean_label(x_train, y_train, param):
    print("Clean Label Attack")
    target_label = param["target_label"]
    num_images = int(param["poisoning_rate"] * y_train.shape[0])

    index = np.where(y_train == target_label)
    index = index[0]
    index = index[:num_images]
    x_train[index] = poison_frequency(x_train[index], y_train[index], param)
    #y_train[index] = target_label
    return x_train


def poison_frequency(x_train, y_train, param):
    if x_train.shape[0] == 0:
        return x_train

    x_train *= 255.
    if param["YUV"]:
        x_train = RGB2YUV(x_train)

    # # transfer to frequency domain
    x_train = DCT(x_train, param["window_size"])  # (idx, ch, w, h)

    # plug trigger frequency
    for i in range(x_train.shape[0]):
        for ch in param["channel_list"]:
            for w in range(0, x_train.shape[2], param["window_size"]):
                for h in range(0, x_train.shape[3], param["window_size"]):
                    for pos in param["pos_list"]:
                        x_train[i][ch][w + pos[0]][h + pos[1]] += param["magnitude"]


    x_train = IDCT(x_train, param["window_size"])  # (idx, w, h, ch)

    if param["YUV"]:
        x_train = YUV2RGB(x_train)

    x_train /= 255.
    x_train = np.clip(x_train, 0, 1)
    return x_train


def impose(x_train, y_train, param):
    x_train = poison_frequency(x_train, y_train, param)
    return x_train
