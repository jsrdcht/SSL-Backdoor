

# SSL-Backdoor

**A unified PyTorch library for backdoor attacks & defenses in self-supervised learning**

[License: MIT](LICENSE.txt)
[Python](pixi.toml)
[PyTorch](pixi.toml)
[GitHub stars](https://github.com/jsrdcht/SSL-Backdoor)


---

SSL-Backdoor is an academic research library for **backdoor attacks in self-supervised learning (SSL)**. Our goal is to provide a comprehensive and unified platform for researchers to implement, evaluate, and compare various attacks and defenses in the context of SSL.

## 📢 News

- **2026-07-07** 🎉 **BadCLIP** attack is now available! Dual-embedding guided backdoor attack on multimodal contrastive learning ([CVPR 2024](https://arxiv.org/abs/2311.12075)).
- **2026-07-07** 🎉 **CLIP-Backdoor** attack is now available! Poisoning-based backdoor attacks on CLIP, based on Carlini et al. ([ICLR 2022](https://openreview.net/forum?id=iC4UHbQ01Mp)).

**Previous updates**

- **2026-02-02** **Decomp** defense is now available! ([ICML 2025](https://openreview.net/forum?id=DWCDyGl6k8))
- **2025-12-02** **SSL-Cleanse** defense is now available! ([ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-73021-4_24))
- **2025-08-11** **DRUPE** attack is now available! ([S&P 2024](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a029/1RjEa5rjsHK))
- **2025-05-19** **DEDE** defense is now available!
- **2025-04-18** **PatchSearch** defense and **BadEncoder** attack are now available!



## Supported Attacks

This library currently supports the following poisoning attack algorithms against SSL models:


| Method           | Paper                                                                                                                                                                                                                    | Venue     | Configs                                                                                                                                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SSL-Backdoor     | [Backdoor attacks on self-supervised learning](https://doi.org/10.1109/CVPR52688.2022.01298)                                                                                                                             | CVPR 2022 | [train](configs/poisoning/sslbkd.yaml) · [test](configs/poisoning/sslbkd_test.yaml)                                                                                                                                                                     |
| BadEncoder       | [BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning](https://ieeexplore.ieee.org/abstract/document/9833644/)                                                                               | S&P 2022  | [config](configs/attacks/badencoder.py) · [train](configs/attacks/badencoder_train.yaml) · [test](configs/attacks/badencoder_test.yaml)                                                                                                                 |
| CTRL             | [An Embarrassingly Simple Backdoor Attack on Self-supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.html) | ICCV 2023 | –                                                                                                                                                                                                                                                       |
| CorruptEncoder   | [Data poisoning based backdoor attacks to contrastive learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html)            | CVPR 2024 | [train](configs/poisoning/corruptencoder_imagenet100.yaml)                                                                                                                                                                                              |
| BLTO (inference) | [Backdoor Contrastive Learning via Bi-level Trigger Optimization](https://openreview.net/forum?id=oxjeePpgSP)                                                                                                            | ICLR 2024 | –                                                                                                                                                                                                                                                       |
| DRUPE            | [Distribution Preserving Backdoor Attack in Self-supervised Learning](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a029/1RjEa5rjsHK)                                                                  | S&P 2024  | [config](configs/attacks/drupe.py) · [train](configs/attacks/drupe_train.yaml) · [test](configs/attacks/badencoder_test.yaml)                                                                                                                           |
| BadCLIP 🆕       | [BadCLIP: Dual-Embedding Guided Backdoor Attack on Multimodal Contrastive Learning](https://arxiv.org/abs/2311.12075)                                                                                                    | CVPR 2024 | [trigger](configs/clip/badclip/optimize_trigger_banana.yaml) · [poison](configs/clip/badclip/poison_badclip_banana_cc3m.yaml) · [train](configs/clip/badclip/train_badclip_banana_cc3m.yaml) · [eval](configs/clip/badclip/eval_zeroshot_imagenet.yaml) |
| CLIP-Backdoor 🆕 | [Poisoning and Backdooring Contrastive Learning](https://openreview.net/forum?id=iC4UHbQ01Mp)                                                                                                                            | ICLR 2022 | [poison](configs/clip/clip_backdoor/poison_sslbkd_banana_cc3m.yaml) · [train](configs/clip/clip_backdoor/clip_vit_b16_cc3m_poisoned.yaml) · [eval](configs/clip/clip_backdoor/eval_zeroshot_imagenet.yaml)                                              |




## Supported Defenses

We are actively developing and integrating defense mechanisms. Currently, the following defenses are implemented:


| Method      | Paper                                                                                                                                                                                                                                      | Venue     | Configs                                                                       |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ----------------------------------------------------------------------------- |
| PatchSearch | [Defending Against Patch-Based Backdoor Attacks on Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Tejankar_Defending_Against_Patch-Based_Backdoor_Attacks_on_Self-Supervised_Learning_CVPR_2023_paper.html) | CVPR 2023 | [doc](./docs/zh_cn/patchsearch.md) · [config](configs/defense/patchsearch.py) |
| SSL-Cleanse | [SSL-Cleanse: Trojan detection and mitigation in self-supervised learning](https://link.springer.com/chapter/10.1007/978-3-031-73021-4_24)                                                                                                 | ECCV 2024 | [config](configs/defense/ssl_cleanse.py)                                      |
| DEDE        | [DeDe: Detecting Backdoor Samples for SSL Encoders via Decoders](http://arxiv.org/abs/2411.16154)                                                                                                                                          | CVPR 2025 | –                                                                             |
| Decomp      | [A Closer Look at Backdoor Attacks on CLIP](https://openreview.net/forum?id=DWCDyGl6k8)                                                                                                                                                    | ICML 2025 | [config](configs/defense/decomp.yaml)                                         |




## Setup

Get started with SSL-Backdoor quickly:

1. **Clone the repository:**
  ```bash
    git clone https://github.com/jsrdcht/SSL-Backdoor.git
    cd SSL-Backdoor
  ```
2. **Environment (Pixi CUDA only):**
  ```bash
    # resolve/create the CUDA environment defined in pixi.toml
    pixi install -e cuda
    # quick check of core deps and CUDA availability
    pixi run -e cuda check
    # open an interactive shell in the CUDA env (optional)
    pixi shell -e cuda
  ```

## Usage



### Training an SSL Model on a Poisoned Dataset

To train an SSL model (e.g., using MoCo v2) with a chosen poisoning attack, you can use the provided scripts. Example for Distributed Data Parallel (DDP) training:

```bash
# Configure your desired attack, SSL method, dataset, etc. in the relevant config file
# (e.g., configs/ssl/moco_config.yaml, configs/poisoning/...)

bash tools/train.sh <path_to_your_config.yaml>
```

*Please refer to the* `configs` *directory and specific training scripts for detailed usage and parameter options.*

# Citation

```bibtex
@misc{jsrdcht_ssl_backdoor_2025,
  title        = {SSL-Backdoor: A PyTorch library for SSL backdoor research},
  author       = {jsrdcht},
  year         = {2025},
  howpublished = {\url{https://github.com/jsrdcht/SSL-Backdoor/}},
  note         = {MIT License, accessed 2025-08-11}
}
```

