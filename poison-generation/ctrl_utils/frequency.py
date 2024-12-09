import  torch
import torch.nn as nn
import kornia
import numpy as np
import cv2
import scipy.fftpack as fftpack

from .image import poison_frequency, DCT, IDCT
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms
from PIL import Image

def tensor_decorator(func):
    def wrapper(self, x_train, *args, **kwargs):
        # Convert PIL Image to Tensor and add batch dimension
        transform = transforms.ToTensor()
        x_train = transform(x_train).unsqueeze(0)

        # Call the original function
        result = func(self, x_train, *args, **kwargs)

        # Remove batch dimension and convert back to PIL Image
        result = result.squeeze(0)
        transform = transforms.ToPILImage()
        result = transform(result)

        return result

    return wrapper

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft


    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))


    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))


    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)


    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)


    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct1(x):
    """
    Discrete Cosine Transform, Type I
    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)




class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""

    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function

        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False  # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)


def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


class PoisonFre():
    def __init__(self, channel_list, window_size, pos_list, lindct=False,  rgb2yuv=False):
        self.channel_list = channel_list
        self.window_size = window_size
        self.pos_list = [(pos_list[0], pos_list[0]), (pos_list[1], pos_list[1])]

        self.lindct = lindct
        self.rgb2yuv = rgb2yuv

        self.xfm = DWTForward(J=3, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
        self.ifm = DWTInverse(mode='zero', wave='haar')

    def RGB2YUV(self, x_rgb):
        """
        x_rgb: B x C x H x W, tensor
        """

        return kornia.color.rgb_to_yuv(x_rgb)

    def YUV2RGB(self, x_yuv):
        """
          x_yuv: B x C x H x W, tensor
          """
        return kornia.color.yuv_to_rgb(x_yuv)

    def DCT(self, x):
        # x: b, ch, h, w
        x_dct = torch.zeros_like(x)
        if not self.lindct:
            for i in range(x.shape[0]):
                for ch in range(x.shape[1]):
                    for w in range(0, x.shape[2], self.window_size):
                        for h in range(0, x.shape[3], self.window_size):
                            # if w + self.window_size > x.shape[2] or h + self.window_size > x.shape[3]:
                            #     sub_dct = dct_2d(x[i][ch][w:, h:], norm='ortho')
                            #     x_dct[i][ch][w:, h:] = sub_dct
                            if h + self.window_size > x.shape[3] and w + self.window_size <= x.shape[2]:
                                x_dct[i][ch][w:w + self.window_size, h:] = x[i][ch][w:w + self.window_size, h:]
                            elif h + self.window_size <= x.shape[3] and w + self.window_size > x.shape[2]:
                                x_dct[i][ch][w:, h:h + self.window_size] = x[i][ch][w:, h:h + self.window_size]
                            elif h + self.window_size > x.shape[3] and w + self.window_size > x.shape[2]:
                                x_dct[i][ch][w:, h:] = x[i][ch][w:, h:]
                            else:
                                sub_dct = dct_2d(x[i][ch][w:w + self.window_size, h:h + self.window_size], norm='ortho')
                                x_dct[i][ch][w:w + self.window_size, h:h + self.window_size] = sub_dct

        else:
            line_dct_2d = lambda x: apply_linear_2d(x, LinearDCT(x.size(1), type='dct', norm='ortho')).data
            for i in range(x.shape[0]):
                for ch in range(x.shape[1]):
                    for w in range(0, x.shape[2], self.window_size):
                        for h in range(0, x.shape[3], self.window_size):
                            # if w + self.window_size > x.shape[2] or h + self.window_size > x.shape[3]:
                            #     sub_dct = line_dct_2d(x[i][ch][w:, h:])
                            #     x_dct[i][ch][w:, h:] = sub_dct
                            if h + self.window_size > x.shape[3] and w + self.window_size <= x.shape[2]:
                                x_dct[i][ch][w:w + self.window_size, h:] = x[i][ch][w:w + self.window_size, h:]
                            elif h + self.window_size <= x.shape[3] and w + self.window_size > x.shape[2]:
                                x_dct[i][ch][w:, h:h + self.window_size] = x[i][ch][w:, h:h + self.window_size]
                            elif h + self.window_size > x.shape[3] and w + self.window_size > x.shape[2]:
                                x_dct[i][ch][w:, h:] = x[i][ch][w:, h:]
                            else:
                                sub_dct = line_dct_2d(x[i][ch][w:w + self.window_size, h:h + self.window_size])
                                x_dct[i][ch][w:w + self.window_size, h:h + self.window_size] = sub_dct

        return x_dct


    def IDCT(self, x,):
        x_idct = torch.zeros_like(x)
        if not self.lindct:
            for i in range(x.shape[0]):
                for ch in range(x.shape[1]):
                    for w in range(0, x.shape[2], self.window_size):
                        for h in range(0, x.shape[3], self.window_size):
                            # if w + self.window_size > x.shape[2] or h + self.window_size > x.shape[3]:
                            #     sub_idct = idct_2d(x[i][ch][w:, h:], norm='ortho')
                            #     x_idct[i][ch][w:, h:] = sub_idct
                            if h + self.window_size > x.shape[3] and w + self.window_size <= x.shape[2]:
                                x_idct[i][ch][w:w + self.window_size, h:] = x[i][ch][w:w + self.window_size, h:]
                            elif h + self.window_size <= x.shape[3] and w + self.window_size > x.shape[2]:
                                x_idct[i][ch][w:, h:h + self.window_size] = x[i][ch][w:, h:h + self.window_size]
                            elif h + self.window_size > x.shape[3] and w + self.window_size > x.shape[2]:
                                x_idct[i][ch][w:, h:] = x[i][ch][w:, h:]
                            else:
                                sub_idct = idct_2d(x[i][ch][w:w + self.window_size, h:h + self.window_size], norm='ortho')
                                x_idct[i][ch][w:w + self.window_size, h:h + self.window_size] = sub_idct

        else:
            line_idct_2d = lambda x: apply_linear_2d(x, LinearDCT(x.size(1), type='idct', norm='ortho')).data
            for i in range(x.shape[0]):
                for ch in range(x.shape[1]):
                    for w in range(0, x.shape[2], self.window_size):
                        for h in range(0, x.shape[3], self.window_size):
                            # if w + self.window_size > x.shape[2] or h + self.window_size > x.shape[3]:
                            #     sub_idct = line_idct_2d(x[i][ch][w:, h:])
                            #     x_idct[i][ch][w:, h:] = sub_idct
                            if h + self.window_size > x.shape[3] and w + self.window_size <= x.shape[2]:
                                x_idct[i][ch][w:w + self.window_size, h:] = x[i][ch][w:w + self.window_size, h:]
                            elif h + self.window_size <= x.shape[3] and w + self.window_size > x.shape[2]:
                                x_idct[i][ch][w:, h:h + self.window_size] = x[i][ch][w:, h:h + self.window_size]
                            elif h + self.window_size > x.shape[3] and w + self.window_size > x.shape[2]:
                                x_idct[i][ch][w:, h:] = x[i][ch][w:, h:]
                            else:
                                sub_idct = line_idct_2d(x[i][ch][w:w + self.window_size, h:h + self.window_size])
                                x_idct[i][ch][w:w + self.window_size, h:h + self.window_size] = sub_idct

        return x_idct


    def DWT(self, x):
        return self.xfm(x)

    def IDWT(self, yl, yh):
        return self.ifm((yl, yh))


    def Poison_Frequency(self, x_train, y_train, poison_list, magnitude):

        if x_train.shape[0] == 0:
            return x_train

        x_train *= 255.


        if self.rgb2yuv:
            x_train = self.RGB2YUV(x_train)



        # transfer to frequency domain
        x_train = self.DCT(x_train)  # (idx, ch, w, h)

        #plug trigger frequency

        for i in range(x_train.shape[0]):
            for ch in self.channel_list:
                for w in range(0, x_train.shape[2], self.window_size):
                    for h in range(0, x_train.shape[3], self.window_size):
                        for pos in poison_list:
                            x_train[i][ch][w + pos[0]][h + pos[1]] += magnitude

        x_train = self.IDCT(x_train)  # (idx, w, h, ch)

        if self.rgb2yuv:
            x_train = self.YUV2RGB(x_train)

        x_train /= 255.
        x_train = torch.clamp(x_train, min =0.0, max= 1.0)
        return x_train, y_train

    @tensor_decorator
    def Poison_Frequency_Diff(self, x_train, magnitude, dwt=False):

        if x_train.shape[0] == 0:
            return x_train

        x_train  = x_train * 255.
        #
        if self.rgb2yuv:
            x_train = self.RGB2YUV(x_train)
        #
        #
        # transfer to frequency domain
        if not dwt:
            x_train = self.DCT(x_train)  # (idx, ch, w, h ）
            #
            for ch in self.channel_list:
                for w in range(0, x_train.shape[2], self.window_size):
                    for h in range(0, x_train.shape[3], self.window_size):
                        for pos in self.pos_list:
                            if w + self.window_size > x_train.shape[2] or h + self.window_size > x_train.shape[3]:
                                continue
                            x_train[:, ch, w+pos[0], h+pos[1]] = x_train[:, ch, w+pos[0], h+pos[1]] + magnitude

            #transfer to time domain
            x_train = self.IDCT(x_train)  # (idx, w, h, ch)
        else:
            yl, yh = self.DWT(x_train)
            yh[-1][ :, 1, -1, :, :]  = yh[-1][ :, 1, -1, :, :] + magnitude
            x_train = self.IDWT(yl, yh)

        if self.rgb2yuv:
            x_train = self.YUV2RGB(x_train)

        x_train /= 255.
        x_train = torch.clamp(x_train, min=0.0, max=1.0)

        return x_train


    def Poison_Celan_Label(self, x_train, y_train, target_class, poison_ratio, pos_list, magnitude):
        poison_num = int(poison_ratio * x_train.shape[0])
        index = np.where(y_train == target_class)[0]
        index = index[:poison_num]
        x_train[index], y_train[index] = self.Poison_Frequency(x_train[index], y_train[index], pos_list, magnitude)

        return x_train, index

    def Poison_Celan_Label_Diff(self, x_train, y_train, target_class, poison_ratio, magnitude, dwt=False, part = True):
        poison_num = int(poison_ratio * x_train.shape[0])
        index = np.where(y_train == target_class)[0]
        if part:
            index = index[:poison_num]
        x_train[index], y_train[index] = self.Poison_Frequency_Diff(x_train[index], y_train[index], magnitude, dwt)

        return x_train, index


class linearRegression(torch.nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(100, 10)

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
   #test_lineardct_3d_cv2()

   param = {
       "dataset": "CIFAR10",  # CIFAR10
       "target_label": 0,  # target label
       "poisoning_rate": 0.04,  # ratio of poisoned samples
       "label_dim": 10,
       "channel_list": [1, 2],  # [0,1,2] means YUV channels, [1,2] means UV channels
       "magnitude": 200,
       "YUV": True,
       "window_size": 32,
       "pos_list": [(31, 31), (15, 15)]
       # "pos_list": [(31, 31), (15, 15), (15, 16), (15, 17),  (16, 15), (16, 16), (16, 17), (17, 15), (17, 16), (17, 17)]
   }

   poisonagent = PoisonFre(32, [1,2], 32, False, True)
   loss = torch.nn.MSELoss()



   x_np = np.random.random(size=(100, 32, 32, 3)).astype(np.float32)
   x_tensor= torch.tensor(x_np).permute(0, 3, 1, 2).cuda() - 0.1


   magnitude = torch.rand(size =(1, 3, 32, 32)).cuda() * 1000
   mask = torch.rand(size = (32, 32))
   # mask = torch

   x_tensor.requires_grad_(True)
   magnitude.requires_grad_(True)

   x_input = x_tensor.clone().detach()

