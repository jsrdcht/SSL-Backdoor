from PIL import Image
import numpy as np

def extract_foreground(image_path, mask_path, output_path):
    # 打开图像和掩码
    image = Image.open(image_path).convert('RGBA')
    mask = Image.open(mask_path).convert('L')  # 灰度图

    # 将掩码转换为二值化图像
    mask_np = np.array(mask)
    mask_np = np.where(mask_np > 0, 255, 0).astype(np.uint8)

    # 将掩码应用于图像的 alpha 通道
    image_np = np.array(image)
    image_np[..., 3] = mask_np  # 设置 alpha 通道为掩码

    # 创建新的图像并保存
    foreground = Image.fromarray(image_np)
    foreground.save(output_path)

# 示例用法
extract_foreground('/workspace/sync/SSL-Backdoor/poison-generation/poisonencoder_utils/references/n01558993/n01558993_31/img.png', '/workspace/sync/SSL-Backdoor/poison-generation/poisonencoder_utils/references/n01558993/n01558993_31/label.png', '/workspace/sync/SSL-Backdoor/poison-generation/poisonencoder_utils/references/n01558993/n01558993_31/output.png')