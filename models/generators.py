'''
Courtsey of: https://github.com/Muzammal-Naseer/Cross-domain-perturbations
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable

###########################
# Generator: Resnet
###########################

# To control feature map in generator
ngf = 64

class GeneratorResnet(nn.Module):
    def __init__(self, inception=False, dim="high"):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorResnet, self).__init__()
        self.inception = inception
        self.dim = dim
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)

        if self.dim == "high":
            self.resblock3 = ResidualBlock(ngf * 4)
            self.resblock4 = ResidualBlock(ngf * 4)
            self.resblock5 = ResidualBlock(ngf * 4)
            self.resblock6 = ResidualBlock(ngf * 4)
        else:
            print("I'm under low dim module!")


        # Input size = 3, n/4, n/4
        self.upsampl1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )


        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

    def forward(self, input):

        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        if self.dim == "high":
            x = self.resblock3(x)
            x = self.resblock4(x)
            x = self.resblock5(x)
            x = self.resblock6(x)
        x = self.upsampl1(x)
        x = self.upsampl2(x)
        x = self.blockf(x)
        if self.inception:
            x = self.crop(x)

        return (torch.tanh(x) + 1) / 2 # Output range [0 1]


class GeneratorAdv(nn.Module):
    def __init__(self, eps=8/255):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(GeneratorAdv, self).__init__()
        self.perturbation = torch.randn(size=(1, 3, 32, 32))
        self.perturbation = nn.Parameter(self.perturbation, requires_grad=True)
        self.eps = eps

    def forward(self, input):
        # perturbation = (torch.tanh(self.perturbation) + 1) / 2
        return input + self.perturbation * self.eps # Output range [0 1]


class Generator_Patch(nn.Module):
    def __init__(self, size=10):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        :param data_dim: for high dimentional dataset (imagenet) 6 resblocks will be add otherwise only 2.
        '''
        super(Generator_Patch, self).__init__()
        self.perturbation = torch.randn(size=(1, 3, size, size))
        self.perturbation = nn.Parameter(self.perturbation, requires_grad=True)

    def forward(self, input):
        # perturbation = (torch.tanh(self.perturbation) + 1) / 2
        random_x = np.random.randint(0, input.shape[-1] - self.perturbation.shape[-1])
        random_y = np.random.randint(0, input.shape[-1] - self.perturbation.shape[-1])
        input[:, :, random_x:random_x + self.perturbation.shape[-1], random_y:random_y + self.perturbation.shape[-1]] = self.perturbation
        return input # Output range [0 1]


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual

if __name__ == '__main__':
    netG = GeneratorResnet()
    test_sample = torch.rand(1, 3, 640, 480)
    My_output = test_sample
    print('Generator output:', netG(test_sample).size())
    print('Generator parameters:', sum(p.numel() for p in netG.parameters() if p.requires_grad)/1000000)