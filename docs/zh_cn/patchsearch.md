# PatchSearch 防御

PatchSearch 对可疑训练集进行特征聚类与补丁搜索，再训练补丁分类器过滤可疑样本。实现位于 `ssl_backdoor/defenses/patchsearch/`，运行入口为 `tools/run_patchsearch.py`。

## 环境与配置

Pixi 环境包含 FAISS CPU 聚类库和 Grad-CAM 1.5。模型计算优先使用 CUDA，无 CUDA 时回退到 CPU；正式数据集建议使用 GPU。

编辑 `configs/defense/patchsearch.py`，至少替换以下信息：

- `weights_path`：待检测编码器的检查点。
- `train_file`：可疑训练集列表，每行格式为 `图像路径 原始类别标签`。
- `dataset_name`：数据集名称，决定归一化参数及过滤阶段的图像大小。
- `output_dir`、`experiment_id`：结果保存位置。

搜索阶段将图像缩放至 256 后中心裁剪为 224 × 224；过滤阶段采用对应数据集的分辨率。两个阶段均使用公共数据集配置中的归一化参数。

支持 ResNet 编码器，以及旧的 `moco_resnet18`、`moco_resnet50` 架构名称。模型加载支持普通权重字典和 `state_dict` 等检查点封装，处理 `module`、`encoder_q`、`base_encoder` 等前缀，并排除投影头。CIFAR 编码器使用 3 × 3 首层卷积及 Identity maxpool；STL10 根据检查点兼容旧的 3 × 3 或标准 7 × 7 首层卷积，保留标准 maxpool。缺失编码器参数会报错。

从仓库根目录运行。先将 `PYTHON_BIN` 设置为已安装依赖的环境中 Python 可执行文件的绝对路径，再执行：

```bash
bash tools/run_patchsearch.sh --config configs/defense/patchsearch.py
```

仅执行搜索阶段：

```bash
bash tools/run_patchsearch.sh --config configs/defense/patchsearch.py --skip_filter
```

可通过 `--output_dir` 和 `--experiment_id` 覆盖配置。`num_workers` 分别在顶层和 `filter` 中设置，小规模验证可设为 0。过滤配置须满足：`topk_poisons` 小于样本数，移除顶部 `top_p` 比例及底部 `topk_poisons` 样本后仍有训练数据。

可选的 `poison_config_path` 用于构建额外的干净与带触发器测试集；基础示例不依赖这一选项。

## 指标

搜索阶段返回毒性分数、样本排序、Top-k 检测准确率、AUROC 和 AUPRC。这里的 AUPRC 采用 **Average Precision** 计算，返回值范围为 0–1，控制台以百分比展示。过滤阶段用毒样本概率计算 AUROC、Average Precision，并报告 Recall、FPR 和 Precision；原有返回格式保持不变。

默认通过图像路径是否包含 `poison` 推断评估用的毒样本标签。这只用于检测指标，不是模型推断出的标签。如果文件命名未表达真实毒样本身份，报告的检测指标不能解释为实际检测效果。单类别测试集无法计算完整的检测指标，程序会提示并按现有约定报告为 0。

## 输出

结果保存在 `output_dir/experiment_id/`：

- `patchsearch.log`：搜索及过滤日志。
- `cached_feats.pth`：特征、原始类别、毒样本标签及样本索引的缓存。
- `poison-scores.npy`：每个样本的毒性分数。
- `sorted_indices.npy`：按毒性分数从高到低排列的索引。
- `all_top_poison_patches/`：从可疑样本提取的候选补丁。
- `poison_classifier_topk_<数量>_ensemble_<数量>_max_iter_<次数>/filtered.txt`：过滤后的训练集列表。

缓存按输出目录复用，不校验模型或数据集版本。更换模型、数据集、预处理或搜索参数时，应使用新的实验目录，避免复用旧结果。
