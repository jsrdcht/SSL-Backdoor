"""
BadEncoder (Backdoor Self-Supervised Learning) 攻击方法的使用示例。
"""

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
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='BadEncoder攻击示例')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--test_config', type=str, required=True,
                        help='测试配置文件路径，支持yaml格式')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='实验ID（可选，优先使用配置文件中的值）')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 1. 加载基础配置
    print(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)
    # 1.2 加载攻击的测试文件配置
    print(f"加载攻击配置文件: {args.test_config}")
    test_config = load_config(args.test_config)
    if not isinstance(test_config, dict):
        raise ValueError(f"攻击配置文件 {args.test_config} 格式错误")
    test_config = argparse.Namespace(**test_config)
    
    # 2. 命令行参数覆盖基础配置
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    
    # 确保必要的参数存在
    required_params = ['pretrained_encoder', 'reference_file', 'trigger_file']
    for param in required_params:
        if param not in config or not config[param]:
            raise ValueError(f"缺少必要参数: {param}，请在配置文件中设置")
    
    # 设置输出目录
    config['output_dir'] = os.path.join(config['output_dir'], config['experiment_id'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建日志文件
    logger_path = os.path.join(config['output_dir'], "log.txt")
    config['logger_file'] = open(logger_path, 'w')

    print("BadEncoder攻击配置:")
    print(f"模型架构: {config.get('arch', 'unknown')}")
    print(f"预训练模型: {config['pretrained_encoder']}")
    print(f"数据集名称: {config.get('dataset_name', 'unknown')}")
    print(f"目标标签: {config.get('reference_label', 'unknown')}")
    print(f"输出目录: {config.get('output_dir', 'unknown')}")
    
    # 3. 创建数据集
    print("加载数据集...")
    
    # 3.1 获取影子数据集（用于训练BadEncoder）
    shadow_dataset, memory_dataset = get_poisoning_dataset(
        argparse.Namespace(**config)
    )
    
    # 3.2 获取评估数据集
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        dataset_params[config['shadow_dataset']]['normalize']
    ])
    
    # 下游任务的训练数据集
    downstream_train_dataset = FileListDataset(
        args=test_config,
        path_to_txt_file=test_config.train_file,
        transform=transform,
    )
    # 下游任务的干净测试数据集
    if hasattr(test_config, 'train_file') and test_config.train_file:
        test_data_clean = FileListDataset(
            args=test_config,
            path_to_txt_file=test_config.test_file,
            transform=transform
        )
    # 创建后门测试数据集
    from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset
    test_data_backdoor = OnlineUniversalPoisonedValDataset(
        args=test_config,
        path_to_txt_file=test_config.test_file,
        transform=transform
    )

    # 5. 创建预训练模型
    # TODO : 这里用的是 DUPRE 的实现，以后换成我的
    # from ssl_backdoor.attacks.drupe.DRUPE.models import get_encoder_architecture_usage
    # clean_model = get_encoder_architecture_usage(argparse.Namespace(**config)).cuda()
    # clean_model.eval()


    # if config['pretrained_encoder'] != '':
    #     print(f'加载预训练编码器: {config["pretrained_encoder"]}')
    #     if config['encoder_usage_info'] == 'cifar10' or config['encoder_usage_info'] == 'stl10':
    #         checkpoint = torch.load(config['pretrained_encoder'])
    #         pretrained_encoder.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.load_state_dict(checkpoint['state_dict'], strict=True)
    #     elif config['encoder_usage_info'] == 'imagenet' or config['encoder_usage_info'] == 'CLIP':
    #         checkpoint = torch.load(config['pretrained_encoder'])
    #         pretrained_encoder.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #         backdoored_model.visual.load_state_dict(checkpoint['state_dict'], strict=True)
    #     else:
    #         raise NotImplementedError(f"未支持的编码器使用信息: {config['encoder_usage_info']}")
    from ssl_backdoor.utils.model_utils import load_model
    clean_model, processor = load_model(config['arch'], config['pretrained_encoder'], dataset=config['encoder_usage_info'])
    clean_model = clean_model.cuda()
    clean_model.eval()

    for p in clean_model.parameters():
        p.requires_grad = True

    
    # 6. 运行BadEncoder攻击
    print("\n====== 开始运行BadEncoder攻击 ======")
    backdoored_model, results = run_badencoder(
        args=argparse.Namespace(**config),
        pretrained_encoder=clean_model,
        shadow_dataset=shadow_dataset,
        memory_dataset=memory_dataset,
        test_data_clean=test_data_clean,
        test_data_backdoor=test_data_backdoor,
        downstream_train_dataset=downstream_train_dataset
    )
    
    # 7. 打印结果摘要
    print("\n====== BadEncoder攻击结果摘要 ======")
    print(f"后门编码器保存路径: {os.path.join(config['output_dir'], 'best_model.pth')}")
    
    if results:
        print("\n====== 下游任务评估 ======")
        print(f"干净测试准确率 (BA): {results['BA']:.2f}%")
        print(f"攻击成功率 (ASR): {results['ASR']:.2f}%")
    
    # 7.1 将后门模型转换为标准格式并保存, 这一步本来是不必要的，但是为了适配 DRUPE的实现，所以需要转换
    encoder_usage = config.get('encoder_usage_info', 'cifar10')

    if encoder_usage in ['cifar10', 'stl10']:
        # 默认 使用的是修改后的 ResNet18 结构，需要转换参数名

        class CustomResNet(ResNet):
            """与 SimCLR 中的 ResNet18 结构保持一致（首层 3×3 卷积）"""

            def __init__(self):
                super().__init__(BasicBlock, [2, 2, 2, 2])
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
                self.maxpool = nn.Identity()
                self.fc = nn.Identity()

        def convert_simclr_state_dict(state_dict):
            """将 SimCLR/BadEncoder 生成的 state_dict 转换为标准 ResNet18 的命名方式"""
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
        print(msg)
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': standard_model.state_dict(),
        }, converted_path)
        print(f"已保存转换后的 ResNet18 权重到: {converted_path}")

    else:
        # 直接保存原始模型，不进行转换
        converted_path = os.path.join(config['output_dir'], 'converted.pth')
        torch.save({
            'state_dict': backdoored_model.state_dict(),
        }, converted_path)
        print(f"已保存原始模型权重到: {converted_path}")

    # 8. 关闭日志文件
    config['logger_file'].close()
    print(f"\n日志保存在: {logger_path}")
    print("可以使用训练好的后门编码器进行下游任务，以验证攻击效果")


if __name__ == '__main__':
    main() 