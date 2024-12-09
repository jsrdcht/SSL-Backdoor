import torchvision.transforms as transforms
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont

def create_patch_image(intensity=30, width=20, gap=40, img_size=(224, 224), original_location=(10, 10)):
    """
    创建一个包含正方形patches的图片以及相应的mask。

    参数:
    - intensity: int, 像素强度值
    - width: int, 每个正方形patch的宽度
    - gap: int, patches之间的距离
    - img_size: tuple, 图片的尺寸 (width, height)
    - original_location: tuple, 最左上角的patch的左上角坐标 (x, y)

    返回:
    - PIL.Image.Image 对象 (原图)
    - PIL.Image.Image 对象 (mask)
    """
    # 创建一张全黑（像素值为0）的图片
    image = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(image)
    
    # 创建一个全黑的mask图片
    mask = Image.new("L", img_size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # 获得original_location的坐标
    orig_x, orig_y = original_location

    # 开始绘制patches
    for i in range(orig_x, img_size[0], width + gap):
        for j in range(orig_y, img_size[1], width + gap):
            # 计算patch的左上角和右下角坐标
            upper_left = (i, j)
            lower_right = (i + width, j + width)
            # 用指定的像素强度值填充这个patch
            draw.rectangle([upper_left, lower_right], intensity)
            
            # 在mask上也绘制一个同样大小的patch，使用255作为像素值
            mask_draw.rectangle([upper_left, lower_right], 255)

    return image, mask

def add_PerilsSSLTrigger(input_image_path,
                    watermark,
                    text=True,
                    fntsize=30,
                    watermark_width=150,
                    position='center',
                    location_min=0.25,
                    location_max=0.75,
                    alpha_composite=True,
                    alpha=0.25,
                    val=False):
    

    val_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224)])
    

    if alpha_composite:
        base_image = Image.open(input_image_path).convert('RGBA')
        if val:
            # preprocess validation images
            base_image = val_transform(base_image)

        img_watermark, watermark_mask = create_patch_image(intensity=30, width=30, gap=30, img_size=base_image.size, original_location=(10, 10))
        img_watermark = img_watermark.convert('RGBA')
        # 将mask转换为NumPy数组
        watermark_mask = np.array(watermark_mask)
        watermark_mask = watermark_mask == 255  # 将255转换为True，0转换为False
        

        # use numpy
        na = np.array(img_watermark).astype(np.float)
        # Halve all alpha values
        # na[..., 3] *=0.5
        transparent = Image.fromarray(na.astype(np.uint8))
        # transparent.show()
        
        # change alpha of base image at corresponding locations
        na = np.array(base_image).astype(np.float)
        # Halve all alpha values
        # location = (max(0, min(location[0], na.shape[1])), max(0, min(location[1], na.shape[0]))) # if location is negative, clip at 0
        # TODO: Aniruddha I ensure that left upper location will never be negative. So I removed clipping.
        
        na[..., 3][watermark_mask] *= alpha
        base_image = Image.fromarray(na.astype(np.uint8))
        # base_image.show()
        transparent = Image.alpha_composite(transparent, base_image)


        transparent = transparent.convert('RGB')
        # transparent.show()
        if val:
            return transparent
        else:
            return transparent, (0,0), (1, 1)
            
    else:                       # pasting            
        img_watermark = Image.open(watermark).convert('RGBA')
        base_image = Image.open(input_image_path)
        # watermark = Image.open(watermark_image_path)
        width, height = base_image.size

        # let's say pasted watermark is 150 pixels wide
        # w_width, w_height = img_watermark.size
        w_width, w_height = watermark_width, int(img_watermark.size[1]*watermark_width/img_watermark.size[0])
        img_watermark = img_watermark.resize((w_width, w_height))                 
        transparent = Image.new('RGBA', (width, height), (0,0,0,0))
        transparent.paste(base_image, (0, 0))
        if position == 'center':
            location = (int((width - w_width)/2), int((height - w_height)/2))
            transparent.paste(img_watermark, location, mask=img_watermark)
        elif position == 'multiple':
            for w in [int(base_image.size[0]*i) for i in [0.25, 0.5, 0.75]]:
                for h in [int(base_image.size[1]*i) for i in [0.25, 0.5, 0.75]]:
                    location = (int(w - w_width/2), int(h - w_height/2))  
                    transparent.paste(img_watermark, location, mask=img_watermark)
        elif position == 'random':
            location = (random.randint(int(base_image.size[0]*0.25 - w_width/2), int(base_image.size[0]*0.75 - w_width/2)), 
                        random.randint(int(base_image.size[1]*0.25 - w_height/2), int(base_image.size[1]*0.75 - w_height/2)))
            transparent.paste(img_watermark, location, mask=img_watermark)
        else:
            logging.info("Invalid position argument")
            return
        
        transparent = transparent.convert('RGB')
        # transparent.show()
        if val:
            return transparent
        else:
            return transparent, location, (w_width, w_height)