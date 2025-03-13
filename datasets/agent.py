import numpy as np
import torch

from models.generators import GeneratorResnet

from scipy.fftpack import dct, idct
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor

class CTRLPoisoningAgent():
    def __init__(self, args):
        self.args = args
        self.channel_list = [1,2]
        self.window_size = getattr(args, 'window_size', 32)
        self.pos_list = [(15,15), (31,31)]
        self.magnitude = getattr(args, 'attack_magnitude', 50) # although the default value is 50 in CTRL paper, it is recommended to use 100 in their github repo


        self.lindct = False


    def apply_poison(self, img):
        assert isinstance(img, Image.Image), "Input must be a PIL image"
        
        img_mode = img.mode
        img = img.convert('RGB')
        
        img, (height, width, _) = np.array(img), np.array(img).shape
        
        img = self.rgb_to_yuv(img)

        valid_height = height - height % self.window_size
        valid_width = width - width % self.window_size

        valid_img = img[:valid_height, :valid_width, :]

        dct_img = self.DCT(valid_img)

        for ch in self.channel_list:
            for w in range(0, dct_img.shape[0], self.window_size):
                for h in range(0, dct_img.shape[1], self.window_size):
                    for pos in self.pos_list:
                        dct_img[w+pos[0], h+pos[1],ch] = dct_img[w+pos[0], h+pos[1],ch] + self.magnitude
            

        #transfer to time domain
        idct_img = self.IDCT(dct_img)

        img[:valid_height, :valid_width, :] = idct_img
        # 确保数据类型为uint8，以兼容PIL图像格式
        
        img = self.yuv_to_rgb(img)
        img = np.uint8(np.clip(img, 0, 255))
        img = Image.fromarray(img)  # 将数组转回PIL图像
        img = img.convert(img_mode)

        return img


    def rgb_to_yuv(self, img):
        """
        Convert a numpy RGB image to the YUV color space.
        """
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B
        V = 0.615 * R - 0.51499 * G - 0.10001 * B
        yuv_img = np.stack((Y, U, V), axis=-1)
        return yuv_img

    def yuv_to_rgb(self, img):
        """
        Convert a numpy YUV image to the RGB color space.
        """
        Y, U, V = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U
        rgb_img = np.stack((R, G, B), axis=-1)
        return rgb_img
    

    def DCT(self, x):
        """
        Apply 2D DCT on a PIL image in windows of specified size.
        """
        x_dct = np.zeros_like(x)
        if not self.lindct:
            for ch in range(x.shape[2]):  # assuming last axis is channel
                for w in range(0, x.shape[0], self.window_size):
                    for h in range(0, x.shape[1], self.window_size):
                        sub_dct = self.dct_2d(x[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_dct[w:w + self.window_size, h:h + self.window_size, ch] = sub_dct
        return x_dct

    def dct_2d(self, x, norm=None):
        """
        Perform the 2-dimensional DCT, Type II.
        """
        X1 = dct(x, norm=norm, axis=0)
        X2 = dct(X1, norm=norm, axis=1)
        return X2
    
    def IDCT(self, dct_image):
        """
        Apply 2D IDCT on a numpy array containing DCT coefficients in windows of specified size.
        """
        if not isinstance(dct_image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        x_idct = np.zeros_like(dct_image)
        if not self.lindct:
            for ch in range(dct_image.shape[2]):  # assuming last axis is channel
                for w in range(0, dct_image.shape[0], self.window_size):
                    for h in range(0, dct_image.shape[1], self.window_size):
                        sub_idct = self.idct_2d(dct_image[w:w + self.window_size, h:h + self.window_size, ch], norm='ortho')
                        x_idct[w:w + self.window_size, h:h + self.window_size, ch] = sub_idct
        return x_idct

    def idct_2d(self, X, norm=None):
        """
        Perform the 2-dimensional inverse DCT, Type III.
        """
        x1 = idct(X, norm=norm, axis=1)
        x2 = idct(x1, norm=norm, axis=0)
        return x2
    
class AdaptivePoisoningAgent():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.net_G = GeneratorResnet().to(self.device)
        self.net_G.load_state_dict(torch.load(args.generator_path, map_location='cpu')["state_dict"], strict=True)


    @torch.no_grad()
    def apply_generatorG(self, netG, img, eps=8/255, eval_G=True):
        if eval_G:
            netG.eval()
        else:
            netG.train()

        with torch.no_grad():
            adv = netG(img)
            adv = torch.min(torch.max(adv, img - eps), img + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
        return adv

    
    def apply_poison(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
    
        if 'imagenet' in self.args.dataset.lower():
            image = image.resize((224, 224))

        # to tensor
        image = torch.tensor(np.array(image), device=self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        adv = self.apply_generatorG(self.net_G, image, eval_G=True)
        adv = adv.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv = (adv * 255).clip(0, 255).astype(np.uint8)
        adv = Image.fromarray(adv)
        return adv