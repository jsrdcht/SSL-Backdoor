"""
DeDe (Decoder-based Detection) 防御方法的使用示例。
"""

import os
import argparse
import torch
from torchvision import transforms
import logging
import builtins
import sys
import shutil
import torch.nn as nn


# BadEncoder 数据集工具
from ssl_backdoor.attacks.badencoder import datasets as badencoder_datasets

# DeDe 检测与可视化
from ssl_backdoor.defenses.dede import run_dede_detection
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.datasets.dataset import FileListDataset, OnlineUniversalPoisonedValDataset
from ssl_backdoor.utils.model_utils import get_backbone_model
from ssl_backdoor.utils.utils import extract_config_by_prefix
from ssl_backdoor.datasets import dataset_params
from ssl_backdoor.defenses.dede.reconstruction import load_decoder, visualize_pairs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# 将内置 print 函数重定向到 logging
logger = logging.getLogger()

def _print_to_logger(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(map(str, args)) + end.rstrip('\n')
    logger.info(message)

builtins.print = _print_to_logger

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DeDe防御示例')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--test_config', type=str, required=True,
                        help='后门攻击测试配置文件路径 (.yaml格式)')
    parser.add_argument('--shadow_config', type=str, required=True,
                        help='后门攻击训练配置文件路径 (.yaml格式)')
    
    return parser.parse_args()



def main():
    """
    主函数
    """
    args = parse_args()
    
    # 1. 加载基础配置
    print(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)

    # === 新增：复制3个配置脚本到输出目录 ===
    config_files_to_copy = [args.config, args.test_config, args.shadow_config]
    # 先临时加载config，获取输出目录
    temp_config = config if isinstance(config, dict) else {}
    output_dir = os.path.join(temp_config.get('output_dir', 'output'), temp_config.get('experiment_id', 'experiment'))
    os.makedirs(output_dir, exist_ok=True)
    for file_path in config_files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy(file_path, os.path.join(output_dir, os.path.basename(file_path)))
        else:
            print(f"警告: 配置文件 {file_path} 不存在，无法复制。")
    # === 复制结束 ===

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
    
    # 3. 命令行参数覆盖基础配置
    
    # 确保必要的参数存在
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("缺少必要参数: weights_path，请在基础配置文件中设置或使用--weights_path参数")
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    # 确保输出目录存在
    os.makedirs(config['output_dir'], exist_ok=True)

    # 在输出目录中添加文件日志处理器
    log_file_path = os.path.join(config['output_dir'], 'run_dede.log')
    # 如果尚未有指向该文件的 FileHandler，则添加
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == os.path.abspath(log_file_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    print("DeDe防御配置:")
    print(f"模型架构: {config.get('arch', 'unknown')}")
    print(f"模型权重: {config['weights_path']}")
    print(f"数据集名称: {config.get('dataset_name', 'unknown')}")
    print(f"输出目录: {config.get('output_dir', 'unknown')}")

    config = argparse.Namespace(**config)



    # 5. 加载可疑模型
    from ssl_backdoor.utils.model_utils import load_model
    _arch_lower = str(config.arch).lower()
    _model_type = 'huggingface' if ('clip' in _arch_lower or 'siglip' in _arch_lower) else 'pytorch'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    suspicious_model, processor = load_model(_model_type, config.arch, config.weights_path, dataset=config.dataset_name, device=device)
    suspicious_model.eval()

    from transformers.modeling_outputs import BaseModelOutputWithPooling
    class VisionModelWrapper(nn.Module):
        def __init__(self, model, model_type='default'):
            super().__init__()
            self.model = model
            self.model_type = model_type.lower()

        def forward(self, x):
            # 针对 HuggingFace CLIP
            if self.model_type == 'clip':
                # 只传 pixel_values
                outputs = self.model.get_image_features(pixel_values=x)
                # 处理不同类型的输出
                if isinstance(outputs, BaseModelOutputWithPooling):  # DINOv2等Transformers模型
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        image_features = outputs.pooler_output
                    # 如果没有pooler_output，使用last_hidden_state的第一个token（CLS token）
                    elif hasattr(outputs, 'last_hidden_state'):
                        image_features = outputs.last_hidden_state[:, 0]
                    else:
                        raise ValueError("No valid feature extraction method found.")
                else:  # 直接返回tensor的模型
                    image_features = outputs

                return image_features
            # 你可以在这里扩展支持其他 HuggingFace 模型
            else:
                # 普通 torch 模型
                return self.model(x)
    
    suspicious_model = VisionModelWrapper(suspicious_model, model_type=config.arch)
    
    # 6. 创建数据集
    if processor is not None:
        def transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
    else:
        assert hasattr(shadow_config_obj, 'shadow_dataset'), "shadow_dataset 未在基础配置文件中设置"
        assert shadow_config_obj.shadow_dataset in dataset_params, f"shadow_dataset 必须是以下之一: {', '.join(dataset_params.keys())}"
        assert 'normalize' in dataset_params[shadow_config_obj.shadow_dataset].keys(), "normalize 未在基础配置文件中设置"

        transform = transforms.Compose([
            transforms.Resize((shadow_config_obj.image_size, shadow_config_obj.image_size)),
            transforms.ToTensor(),
            dataset_params[shadow_config_obj.shadow_dataset]['normalize']
        ])

    # 6.1. 可疑训练数据集
    print("加载可疑训练数据集...")
    # 默认配置
    suspicious_dataset = FileListDataset(
        args=None, 
        path_to_txt_file=shadow_config_obj.shadow_file,
        transform=transform
    )
    # 加载 badencoder 影子数据集
    # shadow_config_obj.shadow_fraction = 1.0 # if 加载badencoder影子数据集，则设置为1.0
    # suspicious_dataset = badencoder_datasets.BadEncoderDatasetAsOneBackdoorOutput(
    #     args=shadow_config_obj,
    #     shadow_file=shadow_config_obj.shadow_file,
    #     reference_file=shadow_config_obj.reference_file,
    #     trigger_file=shadow_config_obj.trigger_file
    # )
    
    
    # 加载内存数据集（用于评估)
    if hasattr(shadow_config_obj, 'memory_file') and shadow_config_obj.memory_file:
        memory_dataset = FileListDataset(
            args=shadow_config_obj,
            path_to_txt_file=shadow_config_obj.memory_file,
            transform=transform
        )
    else:
        memory_dataset = None
    

    # 6.2 测试数据集
    
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


    

    # 7. 运行DeDe检测
    print("\n====== 开始运行DeDe后门检测 ======")
    results, clean_dataset, poisoned_dataset = run_dede_detection(
        args=config,
        suspicious_model=suspicious_model,
        suspicious_dataset=suspicious_dataset,
        memory_dataset=memory_dataset,
        clean_test_dataset=clean_test_dataset,
        poisoned_test_dataset=poisoned_test_dataset
    )
    
    # 8. 打印结果摘要
    print("\n====== DeDe检测结果摘要 ======")
    print(f"使用阈值: {results['threshold']:.4f}")
    print(f"保留干净样本数量: {results['clean_set_size']}")
    print(f"移除有毒样本数量: {results['poisoned_set_size']}")
    print(f"过滤比例: {results['poisoned_set_size'] / (results['clean_set_size'] + results['poisoned_set_size']) * 100:.2f}%")
    
    if 'test_results' in results and results['test_results']:
        print("\n====== 检测性能评估 ======")
        print(f"ROC AUC: {results['test_results']['roc_auc']:.4f}")
        
        print("\n-- 最优阈值检测性能 --")
        print(f"最优阈值: {results['test_results']['optimal_threshold']:.4f}")
        print(f"recall: {results['test_results']['recall']:.4f}")
        print(f"precision: {results['test_results']['precision']:.4f}")
        print(f"overall accuracy: {results['test_results']['overall_accuracy']:.4f}")
        
        print("\n-- 替代阈值检测性能 --")
        print(f"替代阈值 (干净测试误差均值×1.5): {results['test_results']['test_threshold']:.4f}")
        print(f"recall: {results['test_results']['alt_recall']:.4f}")
        print(f"precision: {results['test_results']['alt_precision']:.4f}")
        print(f"overall accuracy: {results['test_results']['alt_overall_accuracy']:.4f}")
    
    print(f"\n过滤后的数据集文件保存在: {os.path.join(config.output_dir, 'filtered_file_list.txt')}")
    print(f"重建误差数据保存在CSV文件中: {os.path.join(config.output_dir, 'training_error_data.csv')} 和 {os.path.join(config.output_dir, 'test_error_data.csv')}")
    print("可以使用此文件重新训练SSL模型以提高鲁棒性")

    # 9. 重建图像可视化（使用独立的重建脚本函数）
    try:
        decoder_model = load_decoder(config, device="cuda")
        visualize_pairs(config, suspicious_model, decoder_model, clean_test_dataset, poisoned_test_dataset, num_pairs=3)
    except Exception as e:
        print(f"重建图像可视化失败: {e}")

if __name__ == '__main__':
    main() 