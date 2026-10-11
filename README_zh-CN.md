# SSL-Backdoor

[English](README.md) | [简体中文](README_zh-CN.md)

**面向自监督学习后门攻击与防御研究的统一 PyTorch 库**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE.txt)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.2–2.4](https://img.shields.io/badge/PyTorch-2.2--2.4-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900.svg?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![已实现的方法](https://img.shields.io/badge/Implemented-8%20Attacks%20%7C%205%20Defenses-8A2BE2.svg?style=flat-square)](#支持的攻击方法)
[![GitHub Stars](https://img.shields.io/github/stars/jsrdcht/SSL-Backdoor?style=flat-square&logo=github)](https://github.com/jsrdcht/SSL-Backdoor/stargazers)

---

SSL-Backdoor 是一个面向**自监督学习（Self-Supervised Learning，SSL）后门攻击**的学术研究库。我们希望为研究者提供统一的平台，用于实现、评估和比较自监督学习中的各类后门攻击与防御方法。

## 📢 最新进展

- **2026-08-06** 🎉 已支持 **BDetCLIP（[ICML 2025](https://arxiv.org/abs/2405.15269)）**和 **Subspace Detection（[ICLR 2026](https://openreview.net/forum?id=Kpij6oOnJl)）**测试时防御方法！
- **2026-07-07** 🎉 已支持 **BadCLIP**！这是一种针对多模态对比学习、由双嵌入引导的后门攻击方法（[CVPR 2024](https://arxiv.org/abs/2311.12075)）。
- **2026-07-07** 🎉 已支持 **CLIP-Backdoor**！实现基于 Carlini 等人的工作，研究针对 CLIP 的数据投毒与后门攻击（[ICLR 2022](https://openreview.net/forum?id=iC4UHbQ01Mp)）。

**历史更新**

- **2026-02-02** 已支持 **Decomp** 防御方法！（[ICML 2025](https://openreview.net/forum?id=DWCDyGl6k8)）
- **2025-08-11** 已支持 **DRUPE** 攻击方法！（[S&P 2024](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a029/1RjEa5rjsHK)）
- **2025-05-19** 已支持 **DEDE** 防御方法！
- **2025-04-18** 已支持 **PatchSearch** 防御方法和 **BadEncoder** 攻击方法！

## 支持的攻击方法

本库目前支持以下针对 SSL 模型的后门攻击方法：

| 方法 | 论文 | 发表会议 | 配置 |
| --- | --- | --- | --- |
| SSL-Backdoor | [Backdoor attacks on self-supervised learning](https://doi.org/10.1109/CVPR52688.2022.01298) | CVPR 2022 | [训练](configs/poisoning/sslbkd.yaml) · [测试](configs/poisoning/sslbkd_test.yaml) |
| BadEncoder | [BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning](https://ieeexplore.ieee.org/abstract/document/9833644/) | S&P 2022 | [配置](configs/attacks/badencoder.py) · [训练](configs/attacks/badencoder_train.yaml) · [测试](configs/attacks/badencoder_test.yaml) |
| CTRL | [An Embarrassingly Simple Backdoor Attack on Self-supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.html) | ICCV 2023 | – |
| CorruptEncoder | [Data poisoning based backdoor attacks to contrastive learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html) | CVPR 2024 | [训练](configs/poisoning/corruptencoder_imagenet100.yaml) |
| BLTO（推理） | [Backdoor Contrastive Learning via Bi-level Trigger Optimization](https://openreview.net/forum?id=oxjeePpgSP) | ICLR 2024 | – |
| DRUPE | [Distribution Preserving Backdoor Attack in Self-supervised Learning](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a029/1RjEa5rjsHK) | S&P 2024 | [配置](configs/attacks/drupe.py) · [训练](configs/attacks/drupe_train.yaml) · [测试](configs/attacks/badencoder_test.yaml) |
| BadCLIP 🆕 | [BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning](https://arxiv.org/abs/2311.12075) | CVPR 2024 | [触发器优化](configs/clip/badclip/optimize_trigger_banana.yaml) · [数据投毒](configs/clip/badclip/poison_badclip_banana_cc3m.yaml) · [训练](configs/clip/badclip/train_badclip_banana_cc3m.yaml) · [评估](configs/clip/badclip/eval_zeroshot_imagenet.yaml) |
| CLIP-Backdoor 🆕 | [Poisoning and Backdooring Contrastive Learning](https://openreview.net/forum?id=iC4UHbQ01Mp) | ICLR 2022 | [数据投毒](configs/clip/clip_backdoor/poison_sslbkd_banana_cc3m.yaml) · [训练](configs/clip/clip_backdoor/clip_vit_b16_cc3m_poisoned.yaml) · [评估](configs/clip/clip_backdoor/eval_zeroshot_imagenet.yaml) |

## 支持的防御方法

我们持续开发和集成后门防御方法。目前已实现：

| 方法 | 论文 | 发表会议 | 配置与文档 |
| --- | --- | --- | --- |
| PatchSearch | [Defending Against Patch-Based Backdoor Attacks on Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Tejankar_Defending_Against_Patch-Based_Backdoor_Attacks_on_Self-Supervised_Learning_CVPR_2023_paper.html) | CVPR 2023 | [文档](docs/zh_cn/patchsearch.md) · [配置](configs/defense/patchsearch.py) |
| DEDE | [DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](http://arxiv.org/abs/2411.16154) | CVPR 2025 | – |
| Decomp | [A Closer Look at Backdoor Attacks on CLIP](https://openreview.net/forum?id=DWCDyGl6k8) | ICML 2025 | [配置](configs/defense/decomp.yaml) |
| BDetCLIP | [Test-Time Multimodal Backdoor Detection by Contrastive Prompting](https://arxiv.org/abs/2405.15269) | ICML 2025 | [文档](ssl_backdoor/defenses/bdetclip/README.md) · [配置](configs/bdetclip) |
| Subspace Detection | [Test-Time Poisoned Sample Detection by Exploiting Shallow Malicious Matching in Backdoored CLIP](https://openreview.net/forum?id=Kpij6oOnJl) | ICLR 2026 | [文档](ssl_backdoor/defenses/subspace_detection/README.md) · [配置](configs/subspace_detection) |

## 环境安装

1. **克隆仓库：**

   ```bash
   git clone https://github.com/jsrdcht/SSL-Backdoor.git
   cd SSL-Backdoor
   ```

2. **创建 Pixi CUDA 环境：**

   ```bash
   # 根据 pixi.toml 解析依赖并创建 CUDA 环境
   pixi install -e cuda
   # 检查核心依赖和 CUDA 是否可用
   pixi run -e cuda check
   # 可选：进入该环境的交互式 Shell
   pixi shell -e cuda
   ```

## 使用方法

典型的使用流程分为两步：

1. **填写配置文件。** 在 [`configs/`](configs) 中选择所需方法的配置示例，填写数据集路径、模型架构和检查点、输出目录，以及该方法的相关参数。

2. **运行对应的 Bash 脚本。** 在配置好的环境中，从仓库根目录执行 `bash tools/<script>.sh`。请查看对应脚本的配置方式：部分脚本通过命令行接收配置路径，部分脚本在内部指定配置路径。

例如，修改 [`configs/bdetclip/sslbkd.yaml`](configs/bdetclip/sslbkd.yaml) 后，可以通过以下命令启动 BDetCLIP 检测：

```bash
bash tools/run_bdetclip.sh configs/bdetclip/sslbkd.yaml
```

**各算法可以独立使用。** 我们共享通用工具，同时保持各方法的输入要求独立，尽量避免将一个算法的输入绑定为另一个算法的输出。用户可以将本库中的单个攻击或防御方法与其他仓库的实现组合使用。

例如，用户可以在其他仓库中训练后门模型，再将模型检查点导入本库运行防御方法。只需配置匹配的模型架构、检查点格式、预处理方式，以及该防御方法需要的数据集或参考资源。待检测模型无需由本库的攻击代码训练。

## 引用

```bibtex
@misc{jsrdcht_ssl_backdoor_2025,
  title        = {SSL-Backdoor: A PyTorch library for SSL backdoor research},
  author       = {jsrdcht},
  year         = {2025},
  howpublished = {\url{https://github.com/jsrdcht/SSL-Backdoor/}},
  note         = {MIT License, accessed 2025-08-11}
}
```
