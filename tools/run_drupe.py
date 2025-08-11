"""
DRUPE (Distribution and Regularity-based Update for backdooring PrEtrained encoders) 攻击方法的使用示例。
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
# 导入新添加的指标记录工具
from ssl_backdoor.attacks.drupe.metric_logger import MetricLogger
from ssl_backdoor.utils.model_utils import load_model  # 新增: 通用模型加载工具


def log_info(message, config=None):
    """
    日志记录工具函数，优先使用日志记录，如果没有日志对象则使用print
    
    Args:
        message: 要记录的信息
        config: 配置对象，如果有logger_file属性则使用它记录日志
    """
    if config is not None and 'logger_file' in config and config['logger_file'] is not None:
        config['logger_file'].write(f"{message}\n")
        config['logger_file'].flush()
    print(message)


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='DRUPE攻击示例')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--test_config', type=str, required=True,
                        help='测试配置文件路径，支持yaml格式')
    parser.add_argument('--metric_log', type=str, default='/workspace/SSL-Backdoor/log.csv',
                        help='指标记录CSV文件路径')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 1. 加载基础配置
    log_info(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)
    # 1.2 加载攻击的测试文件配置
    log_info(f"加载攻击配置文件: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"攻击配置文件 {args.test_config} 格式错误")
    
    
    # 2. 命令行参数覆盖基础配置
    
    # 确保必要的参数存在
    required_params = ['pretrained_encoder', 'reference_file', 'trigger_file', 'mode']
    for param in required_params:
        if param not in config or not config[param]:
            raise ValueError(f"缺少必要参数: {param}，请在配置文件中设置")
    
    # 设置输出目录
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建日志文件
    logger_path = os.path.join(config['output_dir'], "log.txt")
    config['logger_file'] = open(logger_path, 'w')
    
    # 添加指标记录路径
    config['metric_log_path'] = args.metric_log

    log_info("DRUPE攻击配置:", config)
    log_info(f"模型架构: {config.get('arch', 'unknown')}", config)
    log_info(f"预训练模型: {config['pretrained_encoder']}", config)
    log_info(f"数据集名称: {config.get('shadow_dataset', 'unknown')}", config)
    log_info(f"目标标签: {config.get('reference_label', 'unknown')}", config)
    log_info(f"攻击模式: {config.get('mode', 'unknown')}", config)
    log_info(f"输出目录: {config.get('output_dir', 'unknown')}", config)
    log_info(f"指标记录文件: {config.get('metric_log_path', 'unknown')}", config)

    # 把 test_config 中的参数合并到 config 中
    config_obj = argparse.Namespace(**config)
    test_config_obj = argparse.Namespace(**test_config)
    config_obj.test_config_obj = test_config_obj
    
    # 3. 创建数据集
    log_info("加载数据集...", config)
    
    # 3.1 获取
    shadow_dataset, memory_dataset, downstream_train_dataset, test_data_clean, test_data_backdoor = get_dataset(
        config_obj
    ) 

    
    # 5. 创建预训练模型（旧实现已注释）
    # from ssl_backdoor.attacks.drupe.DRUPE.models import get_encoder_architecture_usage
    # clean_model = get_encoder_architecture_usage(config_obj).cuda()
    # clean_model.eval()

    # 5.1 加载预训练权重（旧实现已注释）
    # if config.get('pretrained_encoder'):
    #     log_info(f"加载预训练权重: {config['pretrained_encoder']}", config)
    #     checkpoint = torch.load(config['pretrained_encoder'], map_location='cpu')
    #     state_dict = checkpoint.get('state_dict', checkpoint)
    #     encoder_usage = config.get('encoder_usage_info', 'cifar10')
    #     try:
    #         if encoder_usage in ['imagenet', 'CLIP'] and hasattr(clean_model, 'visual'):
    #             clean_model.visual.load_state_dict(state_dict, strict=True)
    #         else:
    #             clean_model.load_state_dict(state_dict, strict=True)
    #         log_info("预训练权重加载完成", config)
    #     except RuntimeError as e:
    #         log_info(f"加载预训练权重时出错: {e}", config)
    #         raise

    # === 新实现: 统一加载预训练模型（参考 run_dede.py） ===
    clean_model, _ = load_model(config_obj.arch, config_obj.pretrained_encoder, dataset=config_obj.encoder_usage_info)
    clean_model = clean_model.cuda()
    clean_model.eval()
    
    # 6. 运行DRUPE攻击
    log_info("\n====== 开始运行DRUPE攻击 ======", config)
    backdoored_model, results = run_drupe(
        args=config_obj,
        pretrained_encoder=clean_model,
        shadow_dataset=shadow_dataset,
        memory_dataset=memory_dataset,
        test_data_clean=test_data_clean,
        test_data_backdoor=test_data_backdoor,
        downstream_train_dataset=downstream_train_dataset
    )
    
    # 7. 打印结果摘要
    log_info("\n====== DRUPE攻击结果摘要 ======", config)
    log_info(f"后门编码器保存路径: {os.path.join(config['output_dir'], 'best_model.pth')}", config)
    log_info(f"指标记录文件: {config['metric_log_path']}", config)
    
    if results:
        log_info("\n====== 下游任务评估 ======", config)
        log_info(f"干净测试准确率 (BA): {results['BA']:.2f}%", config)
        log_info(f"攻击成功率 (ASR): {results['ASR']:.2f}%", config)
    
    # 7.1 将后门模型转换为标准格式并保存
    encoder_usage = config.get('encoder_usage_info', 'cifar10')

    if encoder_usage in ['cifar10', 'stl10']:
        # SimCLR 使用的是修改后的 ResNet18 结构，需要转换参数名

        class CustomResNet(ResNet):
            """与 SimCLR 中的 ResNet18 结构保持一致（首层 3×3 卷积）"""

            def __init__(self):
                super().__init__(BasicBlock, [2, 2, 2, 2])
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.maxpool = nn.Identity()
                self.fc = nn.Identity()

        def convert_simclr_state_dict(state_dict):
            """将 SimCLR/DRUPE 生成的 state_dict 转换为标准 ResNet18 的命名方式"""
            new_state_dict = {}
            for key, value in state_dict.items():
                # SimCLR 的编码器部分保存在 'f.f.' 前缀下
                if key.startswith('f.f.'):
                    parts = key.split('.')
                    if len(parts) < 3:
                        continue

                    # 处理 conv1
                    if parts[2] == '0':
                        new_key = 'conv1.weight'
                    # 处理 bn1
                    elif parts[2] == '1':
                        param_type = '.'.join(parts[3:])
                        new_key = f'bn1.{param_type}'
                    # 处理 layer1~layer4（对应 3~6）
                    elif parts[2] in ['3', '4', '5', '6']:
                        layer_idx = int(parts[2]) - 2  # 3->layer1, 4->layer2, 5->layer3, 6->layer4
                        remaining = '.'.join(parts[3:])
                        new_key = f'layer{layer_idx}.{remaining}'
                    else:
                        continue

                    new_state_dict[new_key] = value
            return new_state_dict

        # 执行转换并保存
        converted_state_dict = convert_simclr_state_dict(backdoored_model.state_dict())
        standard_model = CustomResNet()
        # 非严格加载，允许部分丢失（如全连接层）
        msg = standard_model.load_state_dict(converted_state_dict, strict=True)
        log_info(msg, config)
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': standard_model.state_dict(),
        }, converted_path)
        log_info(f"已保存转换后的 ResNet18 权重到: {converted_path}", config)

    else:
        # 对于 ImageNet/CLIP 等，直接按原格式保存
        raw_path = os.path.join(config['output_dir'], 'backdoored_model_raw.pth')
        torch.save({'state_dict': backdoored_model.state_dict()}, raw_path)
        log_info(f"已保存原始后门模型权重到: {raw_path}", config)
    
    # 8. 关闭日志文件
    config['logger_file'].close()
    log_info(f"\n日志保存在: {logger_path}")
    log_info("可以使用训练好的后门编码器进行下游任务，以验证攻击效果")


if __name__ == '__main__':
    main() 