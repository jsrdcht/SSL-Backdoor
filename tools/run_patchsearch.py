"""
PatchSearch防御方法的使用示例。
"""

import os
import argparse
import logging
import torch
from argparse import Namespace
from torch.utils.data import DataLoader, ConcatDataset

from ssl_backdoor.defenses.patchsearch import run_patchsearch, run_patchsearch_filter
from ssl_backdoor.ssl_trainers.utils import load_config
from ssl_backdoor.datasets.dataset import OnlineUniversalPoisonedValDataset, FileListDataset
from ssl_backdoor.defenses.patchsearch.utils.dataset import get_transforms


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='PatchSearch防御示例')
    parser.add_argument('--config', type=str, required=True,
                        help='基础配置文件路径，支持.py或.yaml格式')
    # Add optional arguments to override config
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--experiment_id', type=str, help='实验ID')
    parser.add_argument('--skip_filter', action='store_true', help='跳过第二阶段过滤')
    
    return parser.parse_args()


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 1. 加载基础配置（PatchSearch算法配置）
    print(f"加载基础配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 命令行参数覆盖基础配置
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

    # --- 构造外部测试集 (Clean + Poisoned) ---
    external_test_loader = None
    if 'poison_config_path' in config:
        print(f"\n====== 构造外部测试集 (Balanced Clean + Poisoned) ======")
        poison_config_path = config['poison_config_path']
        print(f"加载毒化配置: {poison_config_path}")
        poison_config = load_config(poison_config_path)
        poison_args = Namespace(**poison_config)

        # 确保 dataset_name 一致
        dataset_name = config.get('dataset_name', 'cifar10')
        image_size = 32 if 'cifar' in dataset_name else 96 if 'stl' in dataset_name else 224
        
        # 获取 transforms
        # 注意: OnlineUniversalPoisonedValDataset 需要 transforms 来做 resize 等
        # 这里我们使用 patchsearch 的 get_transforms，通常是 ToTensor + Normalize
        transform = get_transforms(dataset_name, image_size)

        # 1. Clean Test Set
        clean_test_file = poison_config.get('test_file')
        print(f"加载干净测试集: {clean_test_file}")
        # 使用 FileListDataset 加载干净数据 (target=0 for binary classification in filter)
        # 但是 FileListDataset 返回 (img, target) 其中 target 是原始类别
        # 我们需要封装一下或者在 filter 中处理。
        # run_poison_classifier 的 test 函数期望 DataLoader 返回 (path, images, target, is_poisoned, idx)
        # PoisonDataset/ValPoisonDataset 返回这种格式。
        # 这里我们需要构造兼容的 Dataset。
        
        # 实际上，我们可以重用 PoisonDataset 或者类似结构。
        # 但为了简单，我们可以使用 OnlineUniversalPoisonedValDataset 的 'clean' 模式
        
        # 构造 clean_args
        clean_args = Namespace(**poison_config)
        clean_args.attack_algorithm = 'clean' # 强制为 clean
        
        clean_dataset = OnlineUniversalPoisonedValDataset(
            clean_args,
            path_to_txt_file=clean_test_file,
            transform=transform
        )
        # OnlineUniversalPoisonedValDataset 返回 (img, target)。
        # 且在 __getitem__ 中: 
        # if idx in self.poison_idxs: apply_poison
        # 
        # 等等，run_poison_classifier 的 test 函数:
        # for i, (_, images, _, is_poisoned, inds) in enumerate(test_loader):
        # 它解包5个值。
        # 而 OnlineUniversalPoisonedValDataset 默认返回 (img, target) 除非 rich_output=True。
        
        clean_dataset.rich_output = True # 设置 rich_output
        # 但是 OnlineUniversalPoisonedValDataset 的 rich_output 返回 dict。
        # test 函数期望 tuple unpacking。
        
        # 我们必须适配 test 函数的接口。
        # ssl_backdoor/defenses/patchsearch/poison_classifier.py 中的 test 函数:
        # for i, (_, images, _, is_poisoned, inds) in enumerate(test_loader):
        # 这看起来是期望 ValPoisonDataset 的输出: image_path, img, target, is_poisoned, idx
        
        # 我们需要一个 Wrapper Dataset
        class WrapperDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, is_poisoned_flag, offset=0):
                self.dataset = dataset
                self.is_poisoned_flag = is_poisoned_flag
                self.offset = offset
            
            def __getitem__(self, idx):
                # dataset 返回 (img, target) 或者 dict
                # OnlineUniversalPoisonedValDataset 如果 rich_output=False 返回 (img, target)
                res = self.dataset[idx]
                if isinstance(res, dict):
                    img = res['img']
                    # path = res['img_path']
                else:
                    img, _ = res
                    
                # 构造 path (dummy), img, target (dummy), is_poisoned, idx
                return "dummy_path", img, 0, self.is_poisoned_flag, idx + self.offset
            
            def __len__(self):
                return len(self.dataset)

        # 重新构造 Clean Dataset
        clean_dataset = OnlineUniversalPoisonedValDataset(
            clean_args,
            path_to_txt_file=clean_test_file,
            transform=transform
        )
        clean_wrapper = WrapperDataset(clean_dataset, is_poisoned_flag=False)

        # 2. Poisoned Test Set
        poisoned_dataset = OnlineUniversalPoisonedValDataset(
            poison_args,
            path_to_txt_file=clean_test_file, # 使用相同的文件列表，但在加载时投毒
            transform=transform
        )
        poisoned_wrapper = WrapperDataset(poisoned_dataset, is_poisoned_flag=True, offset=len(clean_dataset))
        
        # Combine
        combined_dataset = ConcatDataset([clean_wrapper, poisoned_wrapper])
        
        external_test_loader = DataLoader(
            combined_dataset,
            batch_size=config.get('batch_size', 64),
            shuffle=False,
            num_workers=config.get('num_workers', 8),
            pin_memory=True
        )
        print(f"外部测试集构造完成，总样本数: {len(combined_dataset)} (Clean: {len(clean_dataset)}, Poisoned: {len(poisoned_dataset)})")

    
    # 运行PatchSearch防御，只使用基础配置
    results = run_patchsearch(
        args=config,  # 传递基础配置
        weights_path=config['weights_path'],
        suspicious_dataset=None,  # 传递加载的有毒数据集
        train_file=config['train_file'],
        dataset_name=config.get('dataset_name', 'imagenet100'),
        output_dir=config.get('output_dir', '/tmp'),
        arch=config.get('arch', 'resnet18'),
        num_clusters=config.get('num_clusters', 100),
        window_w=config.get('window_w', 60),
        repeat_patch=config.get('repeat_patch', 1),
        samples_per_iteration=config.get('samples_per_iteration', 2),
        remove_per_iteration=config.get('remove_per_iteration', 0.25),
        prune_clusters=config.get('prune_clusters', True),
        test_images_size=config.get('test_images_size', 1000),
        batch_size=config.get('batch_size', 64),
        topk_thresholds=config.get('topk_thresholds', [5, 10, 20, 50, 100, 500]),
        experiment_id=config.get('experiment_id', 'patchsearch_defense'),
    )
    
    # 打印最有可能的有毒样本
    print("\n最有可能的前10个有毒样本的索引:")
    for i, idx in enumerate(results["sorted_indices"][:10]):
        is_poison = "是" if results["is_poison"][idx] else "否"
        print(f"#{i+1}: 索引 {idx}, 毒性得分 {results['poison_scores'][idx]:.2f}, 实际是否有毒: {is_poison}")
    
    # 如果不跳过过滤步骤，则运行毒药分类器进行过滤
    # 注意：run_patchsearch 内部也会调用 run_patchsearch_filter 如果提供了 args['filter'] 且没有 skip_filter
    # 但是我们修改了 run_patchsearch 的签名来接受 external_test_loader 并传递给 run_patchsearch_filter
    # 所以这里其实不需要再次手动调用 run_patchsearch_filter，除非 run_patchsearch 没调用它。
    
    # 查看 run_patchsearch 代码 (in __init__.py):
    # 它确实只在 args 中有 'filter' 且 !skip_filter 时调用。
    # 我们的 config 有 'filter' key (见 patchsearch.py).
    # 所以 run_patchsearch 会处理一切。
    
    # 这里的代码块 (lines 78-130 in original) 似乎是多余的或者是手动调用的逻辑？
    # 原代码 run_patchsearch 并没有调用 run_patchsearch_filter!
    # 让我再检查一下 __init__.py。
    
    # 在 __init__.py 中，run_patchsearch 函数只返回了 result_dict。并没有调用 run_patchsearch_filter。
    # wait, my previous read of __init__.py showed run_patchsearch logic.
    # Lines 43-172 of __init__.py (run_patchsearch definition).
    # It calculates scores and returns result_dict.
    # It DOES NOT call run_patchsearch_filter.
    
    # So the original tools/run_patchsearch.py manually calls run_patchsearch_filter.
    # My previous thought about "pass it to run_patchsearch_filter" via "run_patchsearch" was based on a misunderstanding or misreading if I thought run_patchsearch called it.
    
    # Looking at __init__.py again (from my Read):
    # run_patchsearch ends at line 172.
    # It does NOT call filter.
    
    # So I must update the MANUAL call to run_patchsearch_filter in tools/run_patchsearch.py.
    
    if not args.skip_filter and 'filter' in config:
        print("\n====== 第二阶段：运行毒药分类器过滤 ======")
        
        # 获取必要的参数
        train_file = config['train_file']
        experiment_dir = results["output_dir"]
        
        # 从config获取filter配置
        filter_config = config.get('filter', {})
        
        # 运行过滤器
        filtered_file_path = run_patchsearch_filter(
            poison_scores_path= os.path.join(experiment_dir, 'poison-scores.npy'),
            train_file=train_file,
            dataset_name=config.get('dataset_name', 'imagenet100'),
            topk_poisons=filter_config.get('topk_poisons', 20),
            top_p=filter_config.get('top_p', 0.10),
            model_count=filter_config.get('model_count', 5),
            max_iterations=filter_config.get('max_iterations', 2000),
            batch_size=filter_config.get('batch_size', 128),
            num_workers=filter_config.get('num_workers', 8),
            lr=filter_config.get('lr', 0.01),
            momentum=filter_config.get('momentum', 0.9),
            weight_decay=filter_config.get('weight_decay', 1e-4),
            print_freq=filter_config.get('print_freq', 10),
            eval_freq=filter_config.get('eval_freq', 50),
            seed=filter_config.get('seed', 42),
            external_test_loader=external_test_loader # 传递我们构造的 loader
        )
        
        # 评估过滤结果
        logger = logging.getLogger('patchsearch')
        if os.path.exists(filtered_file_path):
            # 计算过滤前后的样本数量
            with open(train_file, 'r') as f:
                original_count = len(f.readlines())
            
            with open(filtered_file_path, 'r') as f:
                filtered_count = len(f.readlines())
            
            removed_count = original_count - filtered_count
            removed_percentage = (removed_count / original_count) * 100

            
            logger.info("\n====== 过滤结果统计 ======")
            logger.info(f"原始样本数量: {original_count}")
            logger.info(f"过滤后样本数量: {filtered_count}")
            logger.info(f"移除样本数量: {removed_count}")
            logger.info(f"移除样本百分比: {removed_percentage:.2f}%")
            logger.info(f"过滤后的数据集文件: {filtered_file_path}")
            logger.info(f"可以使用此文件重新训练您的SSL模型以获得更好的鲁棒性")
        else:
            logger.warning(f"警告: 未找到过滤后的文件 {filtered_file_path}")


if __name__ == '__main__':
    main()
