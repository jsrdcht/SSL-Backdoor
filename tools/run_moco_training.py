#!/usr/bin/env python
# 展示如何调用MoCo训练接口

import os
import sys
import argparse
import yaml  # 导入 yaml 库

# 导入训练器接口和配置加载函数
from ssl_trainers.moco.main_moco import get_trainer, load_config


def main():
    parser = argparse.ArgumentParser(description='运行MoCo训练，命令行参数将覆盖配置文件中的同名参数')
    parser.add_argument('--config', type=str, default='configs/ssl/moco.py',
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--attack_config', type=str, default=None, # 新增：攻击配置文件路径
                        help='后门攻击配置文件路径 (.yaml格式)')
    
    # 添加所有可能需要从命令行覆盖的参数
    # !!! 重要：这里的参数名（如 'dataset'）必须与配置文件中的键名完全一致 !!!
    parser.add_argument('--dataset', type=str, default=None, 
                        help='覆盖配置文件中的数据集名称 (键名: dataset)')
    parser.add_argument('--data', type=str, default=None, # 注意：参数名与config键名 'data' 保持一致
                        help='覆盖配置文件中的数据集路径 (键名: data)')
    parser.add_argument('--experiment_id', type=str, default=None, # 注意：参数名与config键名 'experiment_id' 保持一致
                        help='覆盖配置文件中的实验ID (键名: experiment_id)')
    parser.add_argument('--attack_algorithm', type=str, default=None, # 注意：参数名与config键名 'attack_algorithm' 保持一致
                        help='覆盖配置文件中的攻击算法 (键名: attack_algorithm)')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='覆盖配置文件中的训练轮数 (键名: epochs)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置文件中的批次大小 (键名: batch_size)')
    # 可以根据需要添加更多参数...
    
    args = parser.parse_args()
    
    # 1. 加载基础配置
    config = load_config(args.config)
    print(f"已加载基础配置: {args.config}")
    
    # 2. 加载并合并攻击配置 (如果提供了)
    if args.attack_config:
        try:
            attack_config = load_config(args.attack_config) # 使用同一个加载函数
            if isinstance(attack_config, dict):
                print(f"已加载攻击配置: {args.attack_config}")
                # 合并，attack_config 中的值会覆盖 config 中的同名值
                config.update(attack_config)
                print("攻击配置已合并。")
            else:
                print(f"警告：攻击配置文件 {args.attack_config} 未能正确加载为字典，跳过合并。")
        except Exception as e:
            print(f"警告：加载攻击配置文件 {args.attack_config} 时出错: {e}，跳过合并。")

    # 3. 将命令行参数转为字典 (用于覆盖)
    cmd_args_dict = vars(args)
    
    # 4. 使用命令行参数更新最终配置 (忽略None值和配置文件路径参数)
    for key, value in cmd_args_dict.items():
        # 忽略 None 值以及 config 和 attack_config 本身
        if value is not None and key not in ['config', 'attack_config']:
            if key in config and config[key] != value:
                 print(f"配置更新：'{key}' 从 '{config[key]}' 更新为 '{value}' (来自命令行)")
            elif key not in config:
                 print(f"配置新增：'{key}' 设置为 '{value}' (来自命令行)")
            config[key] = value # 覆盖或添加

    print("\n最终使用的配置:")
    # 为了更清晰地打印配置，可以导入 pprint
    try:
        import pprint
        pprint.pprint(config)
    except ImportError:
        print(config)
    print("-"*30)
    
    # 5. 获取训练器 (传入更新后的config字典)
    trainer = get_trainer(config)
    
    # 6. 启动训练
    trainer()


if __name__ == '__main__':
    main() 