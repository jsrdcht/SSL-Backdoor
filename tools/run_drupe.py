"""

example usage.
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.resnet import ResNet, BasicBlock

from ssl_backdoor.attacks.drupe.drupe import run_drupe
from ssl_backdoor.attacks.drupe.datasets import get_dataset
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.utils.utils import extract_config_by_prefix
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.datasets.dataset import FileListDataset
# Metric logging utility imported for experiment records.
from ssl_backdoor.attacks.drupe.metric_logger import MetricLogger
from ssl_backdoor.utils.model_utils import load_model  # Use unified model loader.

def log_info(message, config=None):
    """

        config: Optional config object containing logger_file
    """
    if config is not None and 'logger_file' in config and config['logger_file'] is not None:
        config['logger_file'].write(f"{message}\n")
        config['logger_file'].flush()
    print(message)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='DRUPE example')
    parser.add_argument('--config', type=str, required=True,
                        help='Base config path, supports .py or .yaml')
    parser.add_argument('--test_config', type=str, required=True,
                        help='Test config path, supports yaml')
    parser.add_argument('--metric_log', type=str, default='log.csv',
                        help='Metric CSV output path')
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # 1. Load base config.
    log_info(f"Load base config file: {args.config}")
    config = load_config(args.config)
    # 1.2 Load attack test config.
    log_info(f"Load attack config file: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"Test config {args.test_config} has invalid format")
    
    
    # 2. Apply command-line overrides to base config.
    
    # Validate required keys.
    required_params = ['pretrained_encoder', 'reference_file', 'trigger_file', 'mode']
    for param in required_params:
        if param not in config or not config[param]:
            raise ValueError(f"Missing required parameter: {param}")
    
    # Set output directory.
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create log file.
    logger_path = os.path.join(config['output_dir'], "log.txt")
    config['logger_file'] = open(logger_path, 'w')
    
    # Set metric log path.
    config['metric_log_path'] = args.metric_log

    log_info("DRUPE attack config:", config)
    log_info(f"Model architecture: {config.get('arch', 'unknown')}", config)
    log_info(f"Pretrained model: {config['pretrained_encoder']}", config)
    log_info(f"Dataset name: {config.get('shadow_dataset', 'unknown')}", config)
    log_info(f"Target label: {config.get('reference_label', 'unknown')}", config)
    log_info(f"Attack mode: {config.get('mode', 'unknown')}", config)
    log_info(f"Output directory: {config.get('output_dir', 'unknown')}", config)
    log_info(f"Metric log file: {config.get('metric_log_path', 'unknown')}", config)

    # Merge test_config fields into config for downstream use.
    config_obj = argparse.Namespace(**config)
    test_config_obj = argparse.Namespace(**test_config)
    config_obj.test_config_obj = test_config_obj
    
    # 3. Build datasets.
    log_info("Loading dataset...", config)
    
    # 3.1 Fetch datasets.
    shadow_dataset, memory_dataset, downstream_train_dataset, test_data_clean, test_data_backdoor = get_dataset(
        config_obj
    ) 

    
    # 5. Pretrained model initialization (legacy implementation retained as comments).
    # from ssl_backdoor.attacks.drupe.DRUPE.models import get_encoder_architecture_usage
    # clean_model = get_encoder_architecture_usage(config_obj).cuda()
    # clean_model.eval()

    # 5.1 Legacy checkpoint loading block (kept as comments).
    # if config.get('pretrained_encoder'):

    #     checkpoint = torch.load(config['pretrained_encoder'], map_location='cpu')
    #     state_dict = checkpoint.get('state_dict', checkpoint)
    #     encoder_usage = config.get('encoder_usage_info', 'cifar10')
    #     try:
    #         if encoder_usage in ['imagenet', 'CLIP'] and hasattr(clean_model, 'visual'):
    #             clean_model.visual.load_state_dict(state_dict, strict=True)
    #         else:
    #             clean_model.load_state_dict(state_dict, strict=True)

    #     except RuntimeError as e:

    #         raise

    # New path: load pretrained model using the unified helper.
    _arch_lower = str(config_obj.arch).lower()
    _model_type = 'huggingface' if ('clip' in _arch_lower or 'siglip' in _arch_lower) else 'pytorch'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_model, _ = load_model(_model_type, config_obj.arch, config_obj.pretrained_encoder, dataset=config_obj.encoder_usage_info, device=device)
    # clean_model = clean_model.cuda()
    clean_model.eval()
    
    # 6. Run DRUPE attack.
    log_info("\n====== Starting DRUPE attack ======", config)
    backdoored_model, results = run_drupe(
        args=config_obj,
        pretrained_encoder=clean_model,
        shadow_dataset=shadow_dataset,
        memory_dataset=memory_dataset,
        test_data_clean=test_data_clean,
        test_data_backdoor=test_data_backdoor,
        downstream_train_dataset=downstream_train_dataset
    )
    
    # 7. Print result summary.
    log_info("\n====== DRUPE attack summary ======", config)
    log_info(f"Backdoored encoder saved to: {os.path.join(config['output_dir'], 'best_model.pth')}", config)
        log_info(f"Metric log file: {config['metric_log_path']}", config)
    
    if results:
        log_info("\n====== ====== Downstream task evaluation ======", config)
        log_info(f"Clean test accuracy (BA): {results['BA']:.2f}%", config)
        log_info(f"Attack success rate (ASR): {results['ASR']:.2f}%", config)
    
    # 7.1 Convert and save backdoored model in standard format.
    encoder_usage = config.get('encoder_usage_info', 'cifar10')

    if encoder_usage in ['cifar10', 'stl10']:
        # SimCLR uses a modified ResNet18; convert parameter names accordingly.

        class CustomResNet(ResNet):
            """ResNet18 architecture aligned with SimCLR (3×3 first convolution)."""

            def __init__(self):
                super().__init__(BasicBlock, [2, 2, 2, 2])
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.maxpool = nn.Identity()
                self.fc = nn.Identity()

        def convert_simclr_state_dict(state_dict):
            """Convert SimCLR/DRUPE state_dict keys to standard ResNet18 naming."""
            new_state_dict = {}
            for key, value in state_dict.items():
                # SimCLR encoder parameters use the 'f.f.' prefix.
                if key.startswith('f.f.'):
                    parts = key.split('.')
                    if len(parts) < 3:
                        continue

                    # Map conv1.
                    if parts[2] == '0':
                        new_key = 'conv1.weight'
                    # Map bn1.
                    elif parts[2] == '1':
                        param_type = '.'.join(parts[3:])
                        new_key = f'bn1.{param_type}'
                    # Map layers 1 to 4 (indices 3 to 6).
                    elif parts[2] in ['3', '4', '5', '6']:
                        layer_idx = int(parts[2]) - 2  # 3->layer1, 4->layer2, 5->layer3, 6->layer4
                        remaining = '.'.join(parts[3:])
                        new_key = f'layer{layer_idx}.{remaining}'
                    else:
                        continue

                    new_state_dict[new_key] = value
            return new_state_dict

        # Convert and persist checkpoint.
        converted_state_dict = convert_simclr_state_dict(backdoored_model.state_dict())
        standard_model = CustomResNet()
        # Strict loading with converted keys.
        msg = standard_model.load_state_dict(converted_state_dict, strict=True)
        log_info(msg, config)
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': standard_model.state_dict(),
        }, converted_path)
        log_info(f"Saved converted ResNet18 checkpoint to: {converted_path}", config)

    else:
        # For ImageNet/CLIP style models, save raw checkpoint directly.
        raw_path = os.path.join(config['output_dir'], 'backdoored_model_raw.pth')
        torch.save({'state_dict': backdoored_model.state_dict()}, raw_path)
        log_info(f"Saved raw backdoored model checkpoint to: {raw_path}", config)
    
    # 8. Close logger output file.
    config['logger_file'].close()
    log_info(f"\nLog saved at: {logger_path}")
    log_info("You can use the trained backdoored encoder in downstream tasks to validate attack effect")

if __name__ == '__main__':
    main() 
