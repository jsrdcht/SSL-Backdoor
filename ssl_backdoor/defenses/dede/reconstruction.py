import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# DeDe specific modules
from ssl_backdoor.defenses.dede.decoder_model import DecoderModel
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.utils.model_utils import get_backbone_model
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.attacks.badencoder import datasets as badencoder_datasets


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """反归一化图像张量 (ImageNet 均值方差)"""
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return tensor * std + mean


def save_image_grid(original_images, reconstructed_images, masks, save_path):
    """保存原始图像、重建图像以及 mask 的对比网格"""
    num_images = len(original_images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # 原始图像
        orig_img = denormalize(original_images[i]).cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")

        # 重建图像
        recon_img = denormalize(reconstructed_images[i]).cpu().permute(1, 2, 0).numpy()
        recon_img = np.clip(recon_img, 0, 1)
        axes[i, 1].imshow(recon_img)
        axes[i, 1].axis("off")

        # Mask
        if masks is not None:
            mask_img = masks[i].expand(3, -1, -1).cpu().permute(1, 2, 0).numpy()
            axes[i, 2].imshow(mask_img, cmap="gray")
            axes[i, 2].axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def build_transform(image_size: int, dataset_name: str):
    """根据数据集名称构建默认的图像变换"""
    if dataset_name not in dataset_params:
        raise ValueError(f"数据集 {dataset_name} 未在 dataset_params 中注册，无法构建变换")
    normalize = dataset_params[dataset_name]["normalize"]
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def load_decoder(config, device="cuda"):
    """根据配置加载解码器模型，并自动从 output_dir 加载 best_decoder.pth"""
    checkpoint_path = os.path.join(config.output_dir, "best_decoder.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到解码器权重文件: {checkpoint_path}")

    decoder_model = DecoderModel(
        image_size=config.image_size,
        patch_size=config.patch_size,
        emb_dim=config.emb_dim,
        encoder_layer=config.encoder_layer,
        encoder_head=config.encoder_head,
        decoder_layer=config.decoder_layer,
        decoder_head=config.decoder_head,
        mask_ratio=getattr(config, "test_mask_ratio", config.mask_ratio),
        arch=config.arch,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder_model.load_state_dict(checkpoint["model_state_dict"])
    decoder_model.eval()
    return decoder_model


def visualize_pairs(config, suspicious_model, decoder_model, clean_dataset, poisoned_dataset, num_pairs=3):
    """可视化多组 (干净, 有毒) 图像的重建结果"""
    recon_dir = os.path.join(config.output_dir, "recon_images")
    os.makedirs(recon_dir, exist_ok=True)

    num_pairs = min(num_pairs, len(clean_dataset), len(poisoned_dataset))
    clean_imgs, poisoned_imgs = [], []
    clean_recons, poisoned_recons = [], []
    clean_masks, poisoned_masks = [], []

    with torch.no_grad():
        for i in range(num_pairs):
            clean_img = clean_dataset[i][0] if isinstance(clean_dataset[i], tuple) else clean_dataset[i]
            poisoned_img = poisoned_dataset[i][0] if isinstance(poisoned_dataset[i], tuple) else poisoned_dataset[i]

            clean_img = clean_img.unsqueeze(0).cuda()
            poisoned_img = poisoned_img.unsqueeze(0).cuda()

            clean_feature = suspicious_model(clean_img)
            clean_recon, clean_mask = decoder_model(clean_img, clean_feature)

            poisoned_feature = suspicious_model(poisoned_img)
            poisoned_recon, poisoned_mask = decoder_model(poisoned_img, poisoned_feature)

            clean_imgs.append(clean_img.squeeze(0))
            poisoned_imgs.append(poisoned_img.squeeze(0))
            clean_recons.append(clean_recon.squeeze(0))
            poisoned_recons.append(poisoned_recon.squeeze(0))
            clean_masks.append(clean_mask.squeeze(0))
            poisoned_masks.append(poisoned_mask.squeeze(0))

    # 合并并保存
    all_originals, all_recons, all_masks = [], [], []
    for i in range(num_pairs):
        all_originals.extend([clean_imgs[i], poisoned_imgs[i]])
        all_recons.extend([clean_recons[i], poisoned_recons[i]])
        all_masks.extend([clean_masks[i], poisoned_masks[i]])

    save_path = os.path.join(recon_dir, f"clean_vs_poisoned_{num_pairs}_pairs.png")
    save_image_grid(all_originals, all_recons, all_masks, save_path)
    print(f"保存 {num_pairs} 组干净与有毒样本重建对比图至: {save_path}")


def visualize_dataset(config, suspicious_model, decoder_model, dataset, num_images=6):
    """对数据集中的若干样本进行重建可视化"""
    recon_dir = os.path.join(config.output_dir, "recon_images")
    os.makedirs(recon_dir, exist_ok=True)

    num_images = min(num_images, len(dataset))
    original_images, reconstructed_images, reconstruction_masks = [], [], []

    with torch.no_grad():
        for idx in range(num_images):
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
            img = img.unsqueeze(0).cuda()
            feature = suspicious_model(img)
            recon_img, mask = decoder_model(img, feature)

            original_images.append(img.squeeze(0))
            reconstructed_images.append(recon_img.squeeze(0))
            reconstruction_masks.append(mask.squeeze(0))

            single_path = os.path.join(recon_dir, f"recon_{idx + 1}.png")
            save_image_grid([img.squeeze(0)], [recon_img.squeeze(0)], [mask.squeeze(0)], single_path)

    grid_path = os.path.join(recon_dir, "recon_grid.png")
    save_image_grid(original_images, reconstructed_images, reconstruction_masks, grid_path)
    print(f"已保存 {num_images} 张样本的重建结果至: {grid_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="DeDe 图像重建脚本 (独立版)")
    parser.add_argument("--config", type=str, required=True, help="基础配置文件路径 (.py / .yaml)")
    parser.add_argument("--test_config", type=str, required=True, help="后门攻击测试配置文件路径 (.yaml格式)")
    parser.add_argument("--shadow_config", type=str, required=True, help="后门攻击训练配置文件路径 (.yaml格式)")
    parser.add_argument("--num_pairs", type=int, default=3, help="可视化干净/有毒对比的样本对数")
    parser.add_argument("--num_images", type=int, default=6, help="单数据集重建可视化的样本数量")
    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. 加载基础配置
    print(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 加载攻击配置（仅用于数据加载）
    print(f"加载攻击的测试配置文件: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"攻击测试配置文件 {args.test_config} 格式错误")
    
    print(f"加载攻击的训练配置文件: {args.shadow_config}")
    shadow_config = load_config(args.shadow_config)
    if not isinstance(shadow_config, dict):
        raise ValueError(f"攻击训练配置文件 {args.shadow_config} 格式错误")
    
    # 转为 Namespace 对象
    test_config_obj = argparse.Namespace(**test_config)
    shadow_config_obj = argparse.Namespace(**shadow_config)
    
    # 3. 处理基础配置
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("缺少必要参数: weights_path，请在基础配置文件中设置")
    
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)
    
    config = argparse.Namespace(**config)
    
    # 4. 创建数据转换
    transform = transforms.Compose([
        transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size)),
        transforms.ToTensor(),
        dataset_params[shadow_config_obj.shadow_dataset]['normalize']
    ])
    
    # 5. 加载数据集
    print("加载可疑训练数据集...")
    # 加载 badencoder 影子数据集
    shadow_config_obj.shadow_fraction = 1.0  # if 加载badencoder影子数据集，则设置为1.0
    suspicious_dataset = badencoder_datasets.BadEncoderDatasetAsOneBackdoorOutput(
        args=shadow_config_obj,
        shadow_file=shadow_config_obj.shadow_file,
        reference_file=shadow_config_obj.reference_file,
        trigger_file=shadow_config_obj.trigger_file
    )
    
    # 加载内存数据集（用于评估)
    memory_dataset = None
    if hasattr(shadow_config_obj, 'memory_file') and shadow_config_obj.memory_file:
        memory_dataset = FileListDataset(
            args=shadow_config_obj,
            path_to_txt_file=shadow_config_obj.memory_file,
            transform=transform
        )
    
    print("加载干净测试数据集...")
    clean_test_dataset = FileListDataset(
        args=None,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )
    
    print("加载有毒测试数据集...")
    poisoned_test_dataset = OnlineUniversalPoisonedValDataset(
        args=test_config_obj,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )
    
    # 6. 加载可疑模型
    suspicious_model = get_backbone_model(config.arch, config.weights_path, dataset=config.dataset_name)
    suspicious_model.eval()
    suspicious_model = suspicious_model.to(device="cuda")
    
    # 7. 加载解码器模型
    decoder_model = load_decoder(config, device="cuda")
    decoder_model.eval()
    decoder_model = decoder_model.to(device="cuda")
    
    # 8. 可视化
    visualize_pairs(config, suspicious_model, decoder_model, clean_test_dataset, poisoned_test_dataset, num_pairs=args.num_pairs)
    # visualize_dataset(config, suspicious_model, decoder_model, suspicious_dataset, num_images=args.num_images)


if __name__ == "__main__":
    main() 