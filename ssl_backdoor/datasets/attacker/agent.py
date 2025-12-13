import numpy as np
import torch
import warnings

from .generators import GeneratorResnet

from scipy.fftpack import dct, idct
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
import io
import zipfile
import requests


class CTRLPoisoningAgent():
    def __init__(self, args):
        self.args = args
        self.channel_list = [1, 2]
        self.window_size = getattr(args, 'window_size', 32)
        self.pos_list = [(15, 15), (31, 31)]
        self.magnitude = getattr(args, 'attack_magnitude', 50)  # although the default value is 50 in CTRL paper, it is recommended to use 100 in their github repo

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
                        dct_img[w + pos[0], h + pos[1], ch] = dct_img[w + pos[0], h + pos[1], ch] + self.magnitude

        # transfer to time domain
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


class BadEncoderPoisoningAgent:
    def __init__(self, args):
        warnings.warn(
            "BadEncoderPoisoningAgent is deprecated and will be removed in a future release. "
            "Please use `ssl_backdoor.datasets.utils.add_watermark` (or the BadEncoderDataset path "
            "in `ssl_backdoor.attacks.badencoder.datasets`) instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        self.args = args

        # vanilla badencoder
        trigger_data = np.load(args.trigger_file)
        self.trigger, self.trigger_mask = trigger_data['t'], trigger_data['tm']  # shape为 (1, 32, 32, 3)
        self.trigger, self.trigger_mask = self.trigger.squeeze(), self.trigger_mask.squeeze()
        assert self.trigger.ndim == 3 or self.trigger.ndim == 4 and self.trigger_mask.shape[0] > 1

    def apply_poison(self, img: Image.Image) -> Image.Image:
        # vanilla badencoder
        # 检查输入图像尺寸是否与 trigger 一致，不一致则 resize
        trigger_shape = self.trigger.shape
        if isinstance(img, Image.Image):
            if img.size != (trigger_shape[1], trigger_shape[0]):
                img = img.resize((trigger_shape[1], trigger_shape[0]), Image.BILINEAR)
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            if img.shape[0] != trigger_shape[0] or img.shape[1] != trigger_shape[1]:
                img = Image.fromarray(img)
                img = img.resize((trigger_shape[1], trigger_shape[0]), Image.BILINEAR)
                img = np.array(img)
        else:
            raise ValueError("img must be an instance of Image.Image or np.ndarray")

        assert img.shape[-1] == 3 or img.shape[-1] == 4, "img must be a 3 or 4 channel image"

        backdoored_img = img * self.trigger_mask + self.trigger
        backdoored_img = backdoored_img.astype(np.uint8)
        backdoored_img = Image.fromarray(backdoored_img)

        return backdoored_img


class BadCLIPPoisoningAgent:
    def __init__(self, args):
        self.args = args
        self.trigger_path = args.trigger_path
        self.trigger_size = args.trigger_size
        self.mode = args.mode
        assert self.mode in ['ours_tnature', 'ours_ttemplate', 'vqa', 'blended_kitty', 'blended_banana'], "badclip mode must be one of the following: ours_tnature, ours_ttemplate, vqa, blended_kitty, blended_banana"
        self.position = args.position
        assert self.position in ['middle', 'random'], "position must be one of the following: middle, random"
        assert self.mode == 'ours_tnature' and self.position == 'middle', "only ours_tnature and middle position is supported for now"

        # BadCLIP 的 trigger 植入是在连续空间中进行的，所以需要将 trigger 转换为连续空间
        if self.mode == 'ours_tnature':
            _trigger = Image.open(self.trigger_path).convert('RGB')
            _trigger = _trigger.resize((self.trigger_size, self.trigger_size), Image.BILINEAR)
            _trigger = np.array(_trigger).astype(np.float32) / 255.0
            _trigger = np.clip(_trigger, 0, 1)
            self.trigger = _trigger
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self.image_size = 224
        print(f"set image_size={self.image_size} for badclip poisoning, it is fixed and cannot be changed. If your dataset image size is not 224, please change the image size manually.")

    def apply_poison(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image)
        image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0, 1)

        if self.mode == 'ours_tnature':
            # 将触发器添加到图像中间位置
            trigger = self.trigger
            # 获取图像和触发器尺寸
            img_h, img_w = image.shape[:2]
            trigger_h, trigger_w = trigger.shape[:2]

            # 计算图像中心点
            c_h = int(img_h / 2)
            c_w = int(img_w / 2)

            # 计算触发器左上角的位置
            s_h = int(c_h - trigger_h / 2)
            s_w = int(c_w - trigger_w / 2)

            # 将触发器放在图像中间位置
            image[s_h:s_h + trigger_h, s_w:s_w + trigger_w] = trigger

            # 转换回PIL图像
            image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


class ExternalServicePoisoningAgent:
    """
    通过外部HTTP服务对图像进行投毒/植入。

    约定：
    - 服务接受multipart/form-data：字段名为 'files'（单文件）以及可选的表单字段，如 'secret'。
    - 若仅上传一张图片，服务直接返回PNG字节流；若多张图片，可能返回zip。
    - 示例服务可参考 `StegaStamp-pytorch/stegastamp/server.py` 的 /encode 接口。
    """

    def __init__(self, args):
        self.args = args
        # 读取服务参数（优先外部命名，其次通用命名）
        self.service_url = getattr(args, 'external_service_url', None) or getattr(args, 'service_url', None)
        if not self.service_url:
            raise ValueError("external_service_url 未设置，无法调用外部服务进行投毒")

        # 对于StegaStamp示例服务，使用 `secret` 文本参数
        self.secret = getattr(args, 'external_secret', getattr(args, 'secret', 'Stega!!'))
        self.timeout = getattr(args, 'external_timeout', 30)

        # 规范化编码端点，缺失时默认追加 /encode
        base = self.service_url.rstrip('/')
        if base.endswith('/encode'):
            self.encode_url = base
        else:
            self.encode_url = f"{base}/encode"

    def _image_to_bytes(self, image):
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError("image 必须是图像路径或 PIL.Image")
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return buf.getvalue()

    def _bytes_to_image(self, raw_bytes):
        return Image.open(io.BytesIO(raw_bytes)).convert('RGB')

    def apply_poison(self, image):
        img_bytes = self._image_to_bytes(image)
        files = {
            'files': ('image.png', img_bytes, 'image/png')
        }
        data = {
            'secret': self.secret
        }

        resp = requests.post(self.encode_url, files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()

        content_type = resp.headers.get('Content-Type', '')
        content = resp.content
        if 'application/zip' in content_type:
            with zipfile.ZipFile(io.BytesIO(content), 'r') as zf:
                names = zf.namelist()
                if len(names) == 0:
                    raise RuntimeError('外部服务返回空zip文件')
                with zf.open(names[0]) as f:
                    content = f.read()

        return self._bytes_to_image(content)
