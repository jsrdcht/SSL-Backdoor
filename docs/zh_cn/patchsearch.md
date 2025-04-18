# PatchSearch 防御方法实现文档

本文档介绍了 SSL-Backdoor 库中集成的 PatchSearch 防御方法的实现细节和使用方法。PatchSearch 旨在检测自监督学习（SSL）模型中的后门攻击，特别是基于补丁（Patch-based）的攻击。

本实现参考了官方 PatchSearch 实现，并整合到 SSL-Backdoor 框架中，以便于研究和评估。

## 核心功能

PatchSearch 的核心逻辑位于 `SSL-Backdoor/ssl_backdoor/defenses/patchsearch/` 目录下。主要的入口函数在 `__init__.py` 文件中定义：

1.  **`run_patchsearch`**:
    *   **目的**: 执行 PatchSearch 的第一阶段，通过特征聚类和迭代式补丁搜索来计算每个训练样本的"毒性得分"（Poison Score），识别出最可疑的样本。
    *   **主要参数**:
        *   `args`: 包含额外配置的字典或 Namespace 对象（例如 `poison_label`）。
        *   `model`/`weights_path`: 需要被检测的预训练 SSL 模型或其权重路径。
        *   `train_file`/`suspicious_dataset`: 包含可疑训练样本的文件列表路径或 `Dataset` 对象。
        *   `dataset_name`: 数据集名称 (如 'imagenet100', 'cifar10')。
        *   `output_dir`: 保存结果的输出目录。
        *   `experiment_id`: 实验标识符，用于创建子目录。
        *   `arch`: 模型架构 (如 'resnet18')。
        *   `num_clusters`: 特征聚类的数量。
        *   `use_cached_feats`: 是否使用缓存的特征（加速重复运行）。
        *   `use_cached_poison_scores`: 是否使用缓存的毒性得分（加速重复运行）。
    *   **返回值**: 一个包含检测结果的字典，关键键值包括：
        *   `poison_scores`: 每个样本的毒性得分 NumPy 数组。
        *   `sorted_indices`: 根据毒性得分降序排列的样本索引 NumPy 数组。
        *   `is_poison`: 标记每个样本是否被认为是毒样本的布尔 NumPy 数组（基于文件路径中是否包含 `poison_label`）。
        *   `topk_accuracy`: 在不同 `k` 值下，识别出的 Top-k 样本中毒样本的比例。
        *   `output_dir`: 实际保存结果的目录路径。
        *   `status` (可选): 如果设置了 `use_cached_feats=False` 并且是首次运行，可能会返回 "CACHED_FEATURES"，表示特征已提取并缓存，需要再次运行以进行检测。

2.  **`run_patchsearch_filter`**:
    *   **目的**: 执行 PatchSearch 的第二阶段（可选）。基于第一阶段计算出的毒性得分和提取的可疑补丁，训练一个简单的分类器来进一步过滤潜在的毒样本，并生成一个净化后的数据集文件。
    *   **主要参数**:
        *   `poison_scores`/`poison_scores_path`: 第一阶段输出的毒性得分数组或其 `.npy` 文件路径。
        *   `output_dir`: 输出目录，通常与第一阶段的 `experiment_dir` 相同。
        *   `train_file`: 原始的训练文件路径。
        *   `poison_dir`: 包含从可疑样本中提取的"毒药补丁"图像的目录（通常是 `output_dir/all_top_poison_patches`）。
        *   `topk_poisons`: 选择多少个最高毒性得分的样本用于训练过滤分类器。
        *   `top_p`: 使用原始数据集中多少比例（按毒性得分排序）的数据来训练分类器。
        *   `model_count`: 训练多少个集成模型以提高鲁棒性。
    *   **返回值**: 过滤后生成的新的训练文件路径 (`.txt` 格式)。该文件只包含被分类器认为是"干净"的样本。

## 使用方法

我们提供了一个示例脚本 `SSL-Backdoor/tools/run_patchsearch.py` 来演示如何使用 PatchSearch 防御。

**运行步骤:**

1.  **准备配置文件**:
    *   **基础配置 (`--config`)**: 主要用于配置 PatchSearch 算法本身的参数，如模型权重路径 (`weights_path`)、输出目录 (`output_dir`)、聚类数量 (`num_clusters`)、批处理大小 (`batch_size`)、工作进程数 (`num_workers`) 以及是否使用缓存 (`use_cached_feats`, `use_cached_poison_scores`) 等。此文件也应包含原始（可能被污染）的训练数据文件路径 (`train_file`)。如果需要运行第二阶段过滤，还需要在此配置文件中添加 `filter` 字典来配置相关参数（如 `topk_poisons`, `top_p`, `model_count` 等）。
    *   **攻击配置 (`--attack_config`)**: *可选但推荐*。此配置文件主要用于定义数据集的加载方式，特别是当你的训练数据是按照特定攻击配置（如 SSLBKD、CTRL 等）生成的时候。`run_patchsearch.py` 会尝试从这个配置加载数据集信息。如果 `attack_config` 中设置了 `save_poisons=True`，脚本会智能地将其指向基础配置中的 `train_file` 并禁用 `save_poisons`，以确保加载的是完整的、可能包含毒药的数据集。

2.  **运行脚本**:
    ```bash
    python tools/run_patchsearch.py \
        --config <path_to_your_patchsearch_config.yaml_or_py> \
        --attack_config <path_to_your_attack_config.yaml> \
        [--output_dir <override_output_directory>] \
        [--experiment_id <override_experiment_id>] \
        [--skip_filter] # 可选，如果只想运行第一阶段检测，添加此参数
    ```

**示例流程:**

*   脚本首先加载基础配置和攻击配置。
*   命令行参数会覆盖配置文件中的相应设置。
*   **关键**: 脚本会使用 `attack_config` 中的数据加载设置（结合基础配置中的 `train_file` 和 `batch_size`/`num_workers`）来创建 `suspicious_dataset`。 **注意**: 当前示例脚本中加载 `suspicious_dataset` 的部分被注释掉了 (`# poison_loader = create_data_loader(...)`)，因此默认情况下 `suspicious_dataset` 会是 `None`。`run_patchsearch` 函数内部会处理这种情况，直接从 `train_file` 加载数据集。如果你需要使用 `attack_config` 中定义的复杂数据加载逻辑（例如特定的数据增强），你需要取消注释并调整 `run_patchsearch.py` 中相关的代码。
*   调用 `run_patchsearch` 执行第一阶段检测。
*   如果返回状态是 `CACHED_FEATURES`，提示用户修改基础配置文件设置 `use_cached_feats=True` 后重新运行。
*   打印 Top-10 最可疑样本的信息（索引、毒性得分、是否真实为毒样本）。
*   如果未指定 `--skip_filter` 且基础配置文件中包含 `filter` 配置，则调用 `run_patchsearch_filter` 执行第二阶段过滤。
*   打印过滤统计信息（移除样本数、百分比）和最终生成的干净数据集文件路径。

## 实验结果示例

我们使用此实现对 MoCo v2 模型（在 ImageNet-100 上训练）遭受 SSLBKD 攻击后的情况进行了防御测试。详细的运行日志可以在以下文件中找到：

`SSL-Backdoor/docs/zh_cn/patchsearch.log`

该日志记录了 PatchSearch 运行过程中的关键输出，包括特征提取、聚类、毒性得分计算以及最终的检测准确率等信息，可以作为运行效果的参考。

## 输出文件结构

运行 PatchSearch 后，在指定的 `output_dir/experiment_id/` 目录下会生成以下主要文件和目录：

*   `feats.npy`: 提取的训练集特征（如果 `use_cached_feats=False` 首次运行）。
*   `poison-scores.npy`: 计算得到的每个样本的毒性得分。
*   `sorted_indices.npy`: 根据毒性得分排序后的样本索引。
*   `patchsearch_results.log`: PatchSearch 运行的详细日志。
*   `all_top_poison_patches/`: (如果运行了补丁提取) 包含从最可疑样本中提取出的潜在"毒药补丁"图像。
*   `filter_results/`: (如果运行了第二阶段过滤) 包含过滤阶段的相关文件。
    *   `filtered_train_file.txt`: 过滤后生成的干净训练文件列表。
    *   `poison_classifier.log`: 毒药分类器训练和评估的日志。
    *   ... (可能包含训练的模型权重等)

这个过滤后的 `filtered_train_file.txt` 可以用于重新训练 SSL 模型，以期获得对后门攻击更鲁棒的模型。 