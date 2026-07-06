import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from datasets.dataset import OnlineUniversalPoisonedValDataset, FileListDataset

# Minimal arg container used by dataset helpers.
class Args:
    def __init__(self):
        self.trigger_size = 50
        self.trigger_path = "assets/triggers/trigger_14.png"
        self.trigger_insert = "patch"
        self.return_attack_target = False
        self.attack_target = 0  # Exclude this class label from output
        self.attack_algorithm = "sslbkd"

def process_dataset(input_txt_file, output_dir, config_file):
    """

        config_file (str): Output text config path
    """
    # Prepare output directory.
    os.makedirs(output_dir, exist_ok=True)
    
    # Build transform chain.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # Build dataset helpers.
    args = Args()
    attack_target = args.attack_target

    # Build clean dataset (use FileListDataset for raw inputs).
    clean_dataset = FileListDataset(
        args=args,
        path_to_txt_file=input_txt_file,
        transform=transform
    )
    
    # Build poisoned dataset helper.
    backdoor_dataset = OnlineUniversalPoisonedValDataset(
        args=args,
        path_to_txt_file=input_txt_file,
        transform=transform,
        pre_inject_mode=False
    )
    
    # Open output config file.
    with open(config_file, 'w', encoding='utf-8') as f:
        # Sample counter.
        saved_count = 0
        
        # Iterate over dataset.
        for idx in range(len(clean_dataset)):
            # Read original image and label.
            clean_img, original_label = clean_dataset[idx]
            
            # Skip samples of the excluded attack target class.
            if original_label == attack_target:
                continue
                
            # Generate backdoor variant.
            backdoor_img, _ = backdoor_dataset[idx]
            
            # Save clean image.
            clean_img_name = f"image_{saved_count:06d}_clean.png"
            clean_img_path = os.path.join(output_dir, clean_img_name)
            if isinstance(clean_img, torch.Tensor):
                clean_img = transforms.ToPILImage()(clean_img)
            clean_img.save(clean_img_path)
            f.write(f"{clean_img_path} 0\n")  # 0 for clean
            
            # Save backdoor image.
            backdoor_img_name = f"image_{saved_count:06d}_backdoor.png"
            backdoor_img_path = os.path.join(output_dir, backdoor_img_name)
            if isinstance(backdoor_img, torch.Tensor):
                backdoor_img = transforms.ToPILImage()(backdoor_img)
            backdoor_img.save(backdoor_img_path)
            f.write(f"{backdoor_img_path} 1\n")  # 1 for poisoned
            
            saved_count += 1
            
            # Print progress.
            if saved_count % 50 == 0:
                print(f"Processed {saved_count} clean+poison pairs")

if __name__ == "__main__":
    # Default paths kept relative to project root.
    input_txt_file = "data/ImageNet-100/valset.txt"
    output_dir = "data/backdoor_images"
    config_file = "data/backdoor_config.txt"
    
    # Run preprocessing.
    process_dataset(input_txt_file, output_dir, config_file)
    print("Dataset processing completed!")
