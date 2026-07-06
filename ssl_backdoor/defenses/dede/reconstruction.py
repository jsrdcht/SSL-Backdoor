import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


import torch.nn.functional as F

import random


def _resize_for_dede(img: torch.Tensor, target_size: int, mean, std):
    if img.shape[-1] == target_size:
        return img

    if not torch.is_tensor(mean):
        mean = torch.tensor(mean, device=img.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std, device=img.device)
    mean = mean.view(1, 3, 1, 1)
    std = std.view(1, 3, 1, 1)

    img_denorm = img * std + mean
    img_resized = F.interpolate(img_denorm, size=(target_size, target_size), mode="bilinear", align_corners=False)
    img_norm = (img_resized - mean) / std
    return img_norm


from torchvision import transforms

def _extract_mean_std(transform):
    if isinstance(transform, transforms.Normalize):
        return transform.mean, transform.std
    elif hasattr(transform, 'transforms'):
        for t in transform.transforms:
            res = _extract_mean_std(t)
            if res is not None:
                return res
    # Add support for SkipAugmentationForTensor wrapper (check full_transform)
    elif hasattr(transform, 'full_transform'):
        return _extract_mean_std(transform.full_transform)
    return None

# DeDe specific modules
from ssl_backdoor.defenses.dede.decoder_model import DecoderModel
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.utils.model_utils import get_backbone_model
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.attacks.badencoder import datasets as badencoder_datasets


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Helper function.."""
    device = tensor.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    return tensor * std + mean


def save_image_grid(original_images, reconstructed_images, masks, save_path):
    """Helper function.."""
    num_images = len(original_images)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        
        orig_img = denormalize(original_images[i]).cpu().permute(1, 2, 0).numpy()
        orig_img = np.clip(orig_img, 0, 1)
        axes[i, 0].imshow(orig_img)
        axes[i, 0].axis("off")

        
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




def load_decoder(config, device="cuda"):
    """Helper function.."""
    checkpoint_path = os.path.join(config.output_dir, "best_decoder.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Decoder weight file not found: {checkpoint_path}")

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
    """Helper function.."""
    recon_dir = os.path.join(config.output_dir, "recon_images")
    os.makedirs(recon_dir, exist_ok=True)

    
    ms = _extract_mean_std(getattr(clean_dataset, 'transform', None))
    if ms is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = ms

    
    indices = random.sample(range(len(clean_dataset)), num_pairs)
    clean_imgs, poisoned_imgs = [], []
    clean_recons, poisoned_recons = [], []
    clean_masks, poisoned_masks = [], []

    with torch.no_grad():
        for idx in indices:
            clean_img = clean_dataset[idx][0] if isinstance(clean_dataset[idx], tuple) else clean_dataset[idx]
            poisoned_img = poisoned_dataset[idx][0] if isinstance(poisoned_dataset[idx], tuple) else poisoned_dataset[idx]

            clean_img = clean_img.unsqueeze(0).cuda()
            poisoned_img = poisoned_img.unsqueeze(0).cuda()

            clean_feature = suspicious_model(clean_img)

            clean_img_for_decoder = _resize_for_dede(clean_img, config.image_size, mean, std)
            clean_recon, clean_mask = decoder_model(clean_img_for_decoder, clean_feature)

            poisoned_feature = suspicious_model(poisoned_img)

            poisoned_img_for_decoder = _resize_for_dede(poisoned_img, config.image_size, mean, std)
            poisoned_recon, poisoned_mask = decoder_model(poisoned_img_for_decoder, poisoned_feature)

            clean_imgs.append(clean_img.squeeze(0))
            poisoned_imgs.append(poisoned_img.squeeze(0))
            clean_recons.append(clean_recon.squeeze(0))
            poisoned_recons.append(poisoned_recon.squeeze(0))
            clean_masks.append(clean_mask.squeeze(0))
            poisoned_masks.append(poisoned_mask.squeeze(0))

    
    all_originals, all_recons, all_masks = [], [], []
    for i in range(num_pairs):
        all_originals.extend([clean_imgs[i], poisoned_imgs[i]])
        all_recons.extend([clean_recons[i], poisoned_recons[i]])
        all_masks.extend([clean_masks[i], poisoned_masks[i]])

    save_path = os.path.join(recon_dir, f"clean_vs_poisoned_{num_pairs}_pairs.png")
    save_image_grid(all_originals, all_recons, all_masks, save_path)
    print(f"Saved {num_pairs} clean-vs-poisoned reconstruction pairs to: {save_path}")


def visualize_dataset(config, suspicious_model, decoder_model, dataset, num_images=6):
    """Helper function.."""
    recon_dir = os.path.join(config.output_dir, "recon_images")
    os.makedirs(recon_dir, exist_ok=True)

    
    ms = _extract_mean_std(getattr(dataset, 'transform', None))
    if ms is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean, std = ms

    num_images = min(num_images, len(dataset))
    
    indices = random.sample(range(len(dataset)), num_images)
    original_images, reconstructed_images, reconstruction_masks = [], [], []

    with torch.no_grad():
        for j, idx in enumerate(indices):
            img = dataset[idx][0] if isinstance(dataset[idx], tuple) else dataset[idx]
            img = img.unsqueeze(0).cuda()
            feature = suspicious_model(img)

            img_for_decoder = _resize_for_dede(img, config.image_size, mean, std)
            recon_img, mask = decoder_model(img_for_decoder, feature)

            original_images.append(img.squeeze(0))
            reconstructed_images.append(recon_img.squeeze(0))
            reconstruction_masks.append(mask.squeeze(0))

            single_path = os.path.join(recon_dir, f"recon_{j + 1}.png")
            save_image_grid([img.squeeze(0)], [recon_img.squeeze(0)], [mask.squeeze(0)], single_path)

    grid_path = os.path.join(recon_dir, "recon_grid.png")
    save_image_grid(original_images, reconstructed_images, reconstruction_masks, grid_path)
    print(f"Saved reconstructions for {num_images} samples to: {grid_path}")


class SkipAugmentationForTensor:
    """Wrapper to skip incompatible augmentations if the input is already a tensor (e.g., .pt poisoned images)"""
    def __init__(self, full_transform, tensor_transform=None):
        self.full_transform = full_transform
        self.tensor_transform = tensor_transform

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            # If Tensor, use tensor_transform (if provided), otherwise skip
            if self.tensor_transform:
                return self.tensor_transform(x)
            return x
        # If PIL Image, execute full augmentation chain
        return self.full_transform(x)

def parse_arguments():
    parser = argparse.ArgumentParser(description="DeDe reconstruction script (standalone)")
    parser.add_argument("--config", type=str, required=True, help="Base config path (.py/.yaml)")
    parser.add_argument("--test_config", type=str, required=True, help="Backdoor test config path (.yaml)")
    parser.add_argument("--shadow_config", type=str, required=True, help="Backdoor training config path (.yaml)")
    parser.add_argument("--num_pairs", type=int, default=3, help="Number of clean/poisoned sample pairs to visualize")
    parser.add_argument("--num_images", type=int, default=6, help="Number of samples to visualize for dataset reconstructions")
    return parser.parse_args()


def main():
    args = parse_arguments()

    
    print(f"Loading base config: {args.config}")
    config = load_config(args.config)
    
    
    print(f"Loading backdoor test config: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"Invalid format for attack test config {args.test_config}")
    
    print(f"Loading backdoor training config: {args.shadow_config}")
    shadow_config = load_config(args.shadow_config)
    if not isinstance(shadow_config, dict):
        raise ValueError(f"Invalid format for attack training config {args.shadow_config}")
    
    
    test_config_obj = argparse.Namespace(**test_config)
    shadow_config_obj = argparse.Namespace(**shadow_config)
    
    
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("Missing required parameter: weights_path. Set it in the base config file.")
    
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    config = argparse.Namespace(**config)
    
    
    # Define transforms for PIL images
    transform_pil = transforms.Compose([
        transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size)),
        transforms.ToTensor(),
        dataset_params[shadow_config_obj.shadow_dataset]['normalize']
    ])

    # Define transforms for Tensors (.pt files)
    transform_tensor = transforms.Compose([
        transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size), antialias=True),
        dataset_params[shadow_config_obj.shadow_dataset]['normalize']
    ])

    # Use SkipAugmentationForTensor to handle both formats
    transform = SkipAugmentationForTensor(transform_pil, transform_tensor)

    ms = _extract_mean_std(transform)
    if ms is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print(f"Failed to parse Normalize, using default ImageNet mean/std: {mean}, {std}")
    else:
        mean, std = list(ms[0]), list(ms[1])
        print(f"Parsed Normalize mean/std: {mean}, {std}")
    
    
    print("Loading suspicious training dataset...")
    
    shadow_config_obj.shadow_fraction = 1.0  
    suspicious_dataset = badencoder_datasets.BadEncoderDatasetAsOneBackdoorOutput(
        args=shadow_config_obj,
        shadow_file=shadow_config_obj.shadow_file,
        reference_file=shadow_config_obj.reference_file,
        trigger_file=shadow_config_obj.trigger_file
    )
    
    print("Loading clean test dataset...")
    clean_test_dataset = FileListDataset(
        args=None,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )
    
    print("Loading poisoned test dataset...")
    poisoned_test_dataset = OnlineUniversalPoisonedValDataset(
        args=test_config_obj,
        path_to_txt_file=test_config_obj.test_file,
        transform=transform
    )
    
    
    suspicious_model = get_backbone_model(config.arch, config.weights_path, dataset=config.dataset_name)
    suspicious_model.eval()
    suspicious_model = suspicious_model.to(device="cuda")
    
    
    decoder_model = load_decoder(config, device="cuda")
    decoder_model.eval()
    decoder_model = decoder_model.to(device="cuda")
    
    
    visualize_pairs(config, suspicious_model, decoder_model, clean_test_dataset, poisoned_test_dataset, num_pairs=args.num_pairs)
    # visualize_dataset(config, suspicious_model, decoder_model, suspicious_dataset, num_images=args.num_images)


if __name__ == "__main__":
    main() 