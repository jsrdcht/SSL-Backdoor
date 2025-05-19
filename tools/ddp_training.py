import os
import sys
import argparse
import yaml

# 导入训练器接口和配置加载函数
from ssl_backdoor.ssl_trainers.trainer import get_trainer
from ssl_backdoor.ssl_trainers.utils import load_config

def main():
    parser = argparse.ArgumentParser(description='运行训练，命令行参数将覆盖配置文件中的同名参数')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    parser.add_argument('--attack_config', type=str, default=None, required=True,
                        help='后门攻击配置文件路径 (.yaml格式)')
    parser.add_argument('--test_config', type=str, default=None,
                        help='测试配置文件路径 (.yaml格式)')
    
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

    # 2.5 加载并存储测试配置 (如果提供了)
    if args.test_config:
        try:
            test_config_dict = load_config(args.test_config)
            if isinstance(test_config_dict, dict):
                print(f"已加载测试配置: {args.test_config}")
                config['test_config'] = test_config_dict
            else:
                print(f"警告：测试配置文件 {args.test_config} 未能正确加载为字典，跳过合并。")
        except Exception as e:
            print(f"警告：加载测试配置文件 {args.test_config} 时出错: {e}，跳过合并。")

    print("\n最终使用的训练配置:", config)
    print("\n最终使用的测试配置:", config['test_config'])
    
    # 5. 获取训练器 (传入更新后的config字典)
    trainer = get_trainer(config)
    
    # 6. 准备测试接口 (如果启用了评估)
    eval_frequency = config.get('eval_frequency', 50)
    print(f"eval_frequency: {eval_frequency}, type: {type(eval_frequency)}")
    

    # 7. 启动训练
    trainer = get_trainer(config)
    trainer()


if __name__ == '__main__':
    main() 