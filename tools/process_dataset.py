import os
import sys
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
sys.path.append("/workspace/SSL-Backdoor")
from datasets.dataset import OnlineUniversalPoisonedValDataset, FileListDataset

# 创建一个简单的参数对象
class Args:
    def __init__(self):
        self.trigger_size = 50  # 从linear_probe.sh中获取
        self.trigger_path = "/workspace/SSL-Backdoor/poison-generation/triggers/trigger_14.png"  # 从linear_probe.sh中获取
        self.trigger_insert = "patch"  # 从linear_probe.sh中获取
        self.return_attack_target = False
        self.attack_target = 0  # 这个类别的样本会被排除
        self.attack_algorithm = "sslbkd"  # 从linear_probe.sh中获取

def process_dataset(input_txt_file, output_dir, config_file):
    """
    处理数据集并保存到新的目录
    
    Args:
        input_txt_file (str): 输入的配置文件路径
        output_dir (str): 输出图片保存的目录
        config_file (str): 新生成的配置文件路径
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # 初始化数据集参数
    args = Args()
    attack_target = args.attack_target

    # 初始化干净数据集（使用FileListDataset而不是OnlineUniversalPoisonedValDataset）
    clean_dataset = FileListDataset(
        args=args,
        path_to_txt_file=input_txt_file,
        transform=transform
    )
    
    # 初始化后门数据集
    backdoor_dataset = OnlineUniversalPoisonedValDataset(
        args=args,
        path_to_txt_file=input_txt_file,
        transform=transform,
        pre_inject_mode=False
    )
    
    # 打开新的配置文件
    with open(config_file, 'w', encoding='utf-8') as f:
        # 计数器
        saved_count = 0
        
        # 遍历数据集
        for idx in range(len(clean_dataset)):
            # 获取原始图片和标签
            clean_img, original_label = clean_dataset[idx]
            
            # 跳过attack_target类别的样本
            if original_label == attack_target:
                continue
                
            # 获取后门图片
            backdoor_img, _ = backdoor_dataset[idx]
            
            # 保存干净图片
            clean_img_name = f"image_{saved_count:06d}_clean.png"
            clean_img_path = os.path.join(output_dir, clean_img_name)
            if isinstance(clean_img, torch.Tensor):
                clean_img = transforms.ToPILImage()(clean_img)
            clean_img.save(clean_img_path)
            f.write(f"{clean_img_path} 0\n")  # 0表示干净图像
            
            # 保存后门图片
            backdoor_img_name = f"image_{saved_count:06d}_backdoor.png"
            backdoor_img_path = os.path.join(output_dir, backdoor_img_name)
            if isinstance(backdoor_img, torch.Tensor):
                backdoor_img = transforms.ToPILImage()(backdoor_img)
            backdoor_img.save(backdoor_img_path)
            f.write(f"{backdoor_img_path} 1\n")  # 1表示后门图像
            
            saved_count += 1
            
            # 打印进度
            if saved_count % 50 == 0:
                print(f"已处理 {saved_count} 对图片（干净+后门）")

if __name__ == "__main__":
    # 设置路径
    input_txt_file = "/workspace/SSL-Backdoor/data/ImageNet-100/valset.txt"  # 从linear_probe.sh中获取测试集路径
    output_dir = "/workspace/detect-backdoor-samples-by-neighbourhood/data/backdoor_images"  # 输出目录
    config_file = "/workspace/detect-backdoor-samples-by-neighbourhood/data/backdoor_config.txt"  # 配置文件路径
    
    # 处理数据集
    process_dataset(input_txt_file, output_dir, config_file)
    print("数据集处理完成！") 