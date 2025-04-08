"""
PatchSearch防御方法的使用示例。
"""

import os
import argparse
from ssl_backdoor.defenses.patchsearch import run_patchsearch
from ssl_backdoor.ssl_trainers.trainer import create_data_loader
from ssl_backdoor.ssl_trainers.utils import load_config


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='PatchSearch防御示例')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--attack_config', type=str, required=True,
                        help='后门攻击配置文件路径 (.yaml格式)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（可选，优先使用配置文件中的路径）')
    parser.add_argument('--experiment_id', type=str, default=None,
                        help='实验ID（可选，优先使用配置文件中的值）')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 1. 加载基础配置（PatchSearch算法配置）
    print(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 加载攻击配置（仅用于数据加载）
    print(f"加载攻击配置文件: {args.attack_config}")
    attack_config = load_config(args.attack_config)
    if not isinstance(attack_config, dict):
        raise ValueError(f"攻击配置文件 {args.attack_config} 格式错误")
    
    # 3. 命令行参数覆盖基础配置
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.experiment_id:
        config['experiment_id'] = args.experiment_id
    
    # 确保必要的参数存在
    if 'weights_path' not in config or not config['weights_path']:
        raise ValueError("缺少必要参数: weights_path，请在基础配置文件中设置或使用--weights参数")
    
    print("PatchSearch防御配置:")
    print(f"模型权重: {config['weights_path']}")
    print(f"数据集名称: {config.get('dataset_name', 'unknown')}")
    print(f"输出目录: {config.get('output_dir', '/workspace/SSL-Backdoor/results/defense')}")
    print(f"实验ID: {config.get('experiment_id', 'patchsearch_defense')}")
    
    # 使用create_data_loader加载有毒数据集，只使用attack_config
    print("使用攻击配置加载有毒可疑数据集...")
    if attack_config.get('save_poisons'):
        print("save_poisons is True, 使用 PatchSearch 的训练文件初始化攻击配置的训练文件")
        attack_config['poisons_saved_path'] = config['train_file']
        attack_config['save_poisons'] = False

    attack_config['distributed'] = False # 默认运行在单 GPU 上
    attack_config['workers'] = config['num_workers'] # 默认缺失的配置从 PatchSearch 的配置中获取
    attack_config['batch_size'] = config['batch_size'] # 默认缺失的配置从 PatchSearch 的配置中获取

    print("检查 attack config", attack_config)

    poison_dataset = None
    # 如果想要从训练风格中加载有毒数据集，请取消注释以下代码
    # # 将字典转换为Namespace对象
    # attack_args = argparse.Namespace(**attack_config)
    # poison_loader = create_data_loader(
    #     attack_args  # 传递Namespace对象而不是字典
    # )
    # poison_dataset = poison_loader.dataset
    
    # 运行PatchSearch防御，只使用基础配置
    results = run_patchsearch(
        args=config,  # 传递基础配置
        weights_path=config['weights_path'],
        suspicious_dataset=poison_dataset,  # 传递加载的有毒数据集
        train_file=config['train_file'],
        dataset_name=config.get('dataset_name', 'imagenet100'),
        output_dir=config.get('output_dir', '/tmp'),
        arch=config.get('arch', 'resnet18'),
        num_clusters=config.get('num_clusters', 100),
        window_w=config.get('window_w', 60),
        batch_size=config.get('batch_size', 64),
        use_cached_feats=config.get('use_cached_feats', False),
        use_cached_poison_scores=config.get('use_cached_poison_scores', False),
        experiment_id=config.get('experiment_id', 'patchsearch_defense')
    )
    
    if "status" in results and results["status"] == "CACHED_FEATURES":
        print("\n特征已缓存。请再次运行此脚本，但在基础配置文件中设置use_cached_feats=True以使用缓存的特征。")
        return
    
    # 打印最有可能的有毒样本
    print("\n最有可能的前10个有毒样本的索引:")
    for i, idx in enumerate(results["sorted_indices"][:10]):
        is_poison = "是" if results["is_poison"][idx] else "否"
        print(f"#{i+1}: 索引 {idx}, 毒性得分 {results['poison_scores'][idx]:.2f}, 实际是否有毒: {is_poison}")


if __name__ == '__main__':
    main() 