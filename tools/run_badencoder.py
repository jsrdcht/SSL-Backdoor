"""Example entry script for BadEncoder (Backdoor Self-Supervised Learning)."""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock

from ssl_backdoor.attacks.badencoder.badencoder import run_badencoder
from ssl_backdoor.attacks.badencoder.datasets import get_poisoning_dataset
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.utils.utils import extract_config_by_prefix
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.datasets.dataset import FileListDataset

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='BadEncoder example')
    parser.add_argument('--config', type=str, required=True,
                        help='Base config path, supports .py or .yaml')
    parser.add_argument('--test_config', type=str, required=True,
                        help='Test config path, supports yaml')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='Optional experiment_id (base config value has higher priority)')
    
    return parser.parse_args()

def main():
    """
    Main workflow.
    """
    args = parse_args()
    
    # 1. Load base config
    print(f"Load base config file: {args.config}")
    config = load_config(args.config)
    # 1.2 Load attack test config
    print(f"Load attack config file: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"Test config {args.test_config} has invalid format")
    test_config = argparse.Namespace(**test_config)
    
    # 2. Apply command-line overrides to base config
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    
    # Validate required parameters.
    required_params = ['pretrained_encoder', 'reference_file', 'trigger_file']
    for param in required_params:
        if param not in config or not config[param]:
            raise ValueError(f"Missing required parameter: {param}, please set it in config")
    
    # Configure output directory
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create logger output file
    logger_path = os.path.join(config['output_dir'], "log.txt")
    config['logger_file'] = open(logger_path, 'w')

    print("BadEncoder attack config:")
    print(f"Model architecture: {config.get('arch', 'unknown')}")
    print(f"Pretrained model: {config['pretrained_encoder']}")
    print(f"Dataset name: {config.get('dataset_name', 'unknown')}")
    print(f"Target label: {config.get('reference_label', 'unknown')}")
    print(f"Output directory: {config.get('output_dir', 'unknown')}")
    
    # 3. Build datasets
    print("Loading dataset...")
    
    # 3.1 Load shadow dataset for BadEncoder training
    shadow_dataset, memory_dataset = get_poisoning_dataset(
        argparse.Namespace(**config)
    )
    
    # 3.2 Build evaluation datasets
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        dataset_params[config['shadow_dataset']]['normalize']
    ])
    
    # Downstream training dataset
    downstream_train_dataset = FileListDataset(
        args=test_config,
        path_to_txt_file=test_config.train_file,
        transform=transform,
    )
    # Downstream clean test dataset
    if hasattr(test_config, 'train_file') and test_config.train_file:
        test_data_clean = FileListDataset(
            args=test_config,
            path_to_txt_file=test_config.test_file,
            transform=transform
        )
    # Poisoned test dataset
    from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
    test_data_backdoor = OnlineUniversalPoisonedValDataset(
        args=test_config,
        path_to_txt_file=test_config.test_file,
        transform=transform
    )

    # 5. Load base model
    # TODO: Current implementation uses DUPRE modules; replace with custom one as needed.
    # from ssl_backdoor.attacks.drupe.DRUPE.models import get_encoder_architecture_usage
    # clean_model = get_encoder_architecture_usage(argparse.Namespace(**config)).cuda()
    # clean_model.eval()

    # if config['pretrained_encoder'] != '':

    #     if config['encoder_usage_info'] == 'cifar10' or config['encoder_usage_info'] == 'stl10':
    #         checkpoint = torch.load(config['pretrained_encoder'])
    #         pretrained_encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #     elif config['encoder_usage_info'] == 'imagenet' or config['encoder_usage_info'] == 'CLIP':
    #         checkpoint = torch.load(config['pretrained_encoder'])
    #         pretrained_encoder.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #     else:

    from ssl_backdoor.utils.model_utils import load_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_model, processor = load_model('pytorch', config['arch'], config['pretrained_encoder'], dataset=config['encoder_usage_info'], device=device)
    # clean_model = clean_model.cuda()
    clean_model.eval()

    for p in clean_model.parameters():
        p.requires_grad = True

    
    # 6. Run BadEncoder attack
    print("\n====== ====== Starting BadEncoder attack ======")
    backdoored_model, results = run_badencoder(
        args=argparse.Namespace(**config),
        pretrained_encoder=clean_model,
        shadow_dataset=shadow_dataset,
        memory_dataset=memory_dataset,
        test_data_clean=test_data_clean,
        test_data_backdoor=test_data_backdoor,
        downstream_train_dataset=downstream_train_dataset
    )
    
    # 7. Print summary
    print("\n====== BadEncoder====== Attack summary ======")
    print(f"Backdoored encoder saved to: {os.path.join(config['output_dir'], 'best_model.pth')}")
    
    if results:
        print("\n====== ====== Downstream task evaluation ======")
        print(f"Clean test accuracy (BA): {results['BA']:.2f}%")
        print(f"Attack success rate (ASR): {results['ASR']:.2f}%")
    
    # 7.1 Convert backdoored model to standard format for DRUPE compatibility.
    encoder_usage = config.get('encoder_usage_info', 'cifar10')

    if encoder_usage in ['cifar10', 'stl10']:
        # The default uses a modified ResNet18; rename keys before saving.

        class CustomResNet(ResNet):
            """Match the SimCLR-style ResNet18 layout (3x3 stem convolution)."""

            def __init__(self):
                super().__init__(BasicBlock, [2, 2, 2, 2])
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.maxpool = nn.Identity()
                self.fc = nn.Identity()

        def convert_simclr_state_dict(state_dict):
            """Convert SimCLR/BadEncoder state_dict keys into standard ResNet18 names."""
            new_state_dict = {}
            for key, value in state_dict.items():
                # SimCLR encoder weights use the 'f.f.' prefix.
                if key.startswith('f.f.'):
                    parts = key.split('.')
                    if len(parts) < 3:
                        continue

                    # conv1 mapping
                    if parts[2] == '0':
                        new_key = 'conv1.weight'
                    # bn1 mapping
                    elif parts[2] == '1':
                        param_type = '.'.join(parts[3:])
                        new_key = f'bn1.{param_type}'
                    # layer1-layer4 mapping (keys 3-6 map to layer1-4)
                    elif parts[2] in ['3', '4', '5', '6']:
                        layer_idx = int(parts[2]) - 2  # 3->layer1, 4->layer2, 5->layer3, 6->layer4
                        remaining = '.'.join(parts[3:])
                        new_key = f'layer{layer_idx}.{remaining}'
                    else:
                        continue

                    new_state_dict[new_key] = value
            return new_state_dict

        # Convert and save model checkpoint.
        converted_state_dict = convert_simclr_state_dict(backdoored_model.state_dict())
        standard_model = CustomResNet()
        # Strict load is enabled; missing FC keys are allowed explicitly.
        msg = standard_model.load_state_dict(converted_state_dict, strict=True)
        print(msg)
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': standard_model.state_dict(),
        }, converted_path)
        print(f"Saved converted ResNet18 checkpoint to: {converted_path}")

    else:
        # Save original model for all other encoder backends.
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': backdoored_model.state_dict(),
        }, converted_path)
        print(f"Saved original backdoored model to: {converted_path}")

    # 8. Close logger file
    config['logger_file'].close()

if __name__ == '__main__':
    main() 
