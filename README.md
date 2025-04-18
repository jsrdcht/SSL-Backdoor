<p align="center">
  <img src="assets/image.png" alt="SSL-Backdoor Logo" width="200"/>
</p>

<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  </a>
  <a href="https://github.com/jsrdcht/SSL-Backdoor/stargazers">
    <img src="https://img.shields.io/github/stars/jsrdcht/SSL-Backdoor?style=social" alt="GitHub Stars">
  </a>
  <img src="https://img.shields.io/badge/python-3.7%2B-blue" alt="Python Version">
</p>

# SSL-Backdoor

SSL-Backdoor is an academic research library dedicated to exploring **poisoning attacks in self-supervised learning (SSL)**. Our goal is to provide a comprehensive and unified platform for researchers to implement, evaluate, and compare various attack and defense mechanisms in the context of SSL.

This library originated as a rewrite of the SSLBKD implementation, ensuring **consistent training protocols** and **hyperparameter fidelity** for fair comparisons. We've since expanded its capabilities significantly.

**Key Features:**

1.  **Unified Poisoning & Training Framework:** Streamlined pipeline for applying diverse poisoning strategies and training SSL models.
2.  **Decoupled Design:** We strive to maintain a decoupled design, allowing each method to be modified independently, while unifying the implementation of essential tools where necessary.

*Future plans include support for multimodal contrastive learning models.*

## 📢 What's New?

✅ **2024-04-18 Update:**

* **PatchSearch defense is now implemented and available!**
* **BadEncoder attack is now implemented and available!**

🔄 **Active Refactoring Underway!** We are currently refactoring the codebase to improve code quality, maintainability, and ease of use. Expect ongoing improvements!

✅ **Current Support:**

*   **Attack Algorithms:** SSLBKD, CTRL, CorruptEncoder, BLTO (inference only), BadEncoder
*   **SSL Methods:** MoCo, SimCLR, SimSiam, BYOL

🛡️ **Current Defenses:**

*   **PatchSearch**

Stay tuned for more updates!

## Supported Attacks

This library currently supports the following poisoning attack algorithms against SSL models:

| Aliase       | Paper                                                                                                                                                              | Conference | Config |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------|
| SSLBKD          | [Backdoor attacks on self-supervised learning](https://doi.org/10.1109/CVPR52688.2022.01298)                                                                              | CVPR 2022  | [config](configs/poisoning/poisoning_based/sslbkd.yaml) |
| CTRL            | [An Embarrassingly Simple Backdoor Attack on Self-supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.html) | ICCV 2023  |  |
| CorruptEncoder  | [Data poisoning based backdoor attacks to contrastive learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html)       | CVPR 2024  |  |
| BLTO (inference)| [BACKDOOR CONTRASTIVE LEARNING VIA BI-LEVEL TRIGGER OPTIMIZATION](https://openreview.net/forum?id=oxjeePpgSP)                                                              | ICLR 2024  |  |
| BadEncoder | [BadEncoder: Backdoor Attacks to Pre-trained Encoders in Self-Supervised Learning](https://ieeexplore.ieee.org/abstract/document/9833644/) | S&P 2022|  |

## Supported Defenses

We are actively developing and integrating defense mechanisms. Currently, the following defense is implemented:

| Aliase        | Paper                                                                                                      | Config                         | Conference         |
|------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------|----------------|
| PatchSearch    | [Defending Against Patch-Based Backdoor Attacks on Self-Supervised Learning](https://openaccess.thecvf.com/content/CVPR2023/html/Tejankar_Defending_Against_Patch-Based_Backdoor_Attacks_on_Self-Supervised_Learning_CVPR_2023_paper.html)                                       | [doc](./docs/zh_cn/patchsearch.md), [config](configs/defense/patchsearch.py) | CVPR2023       |

## Setup

Get started with SSL-Backdoor quickly:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jsrdcht/SSL-Backdoor.git
    cd SSL-Backdoor
    ```

2.  **Install dependencies (optional but recommended):**
    ```bash
    pip install -r requirements.txt
    ```
    *Consider using a virtual environment (`conda`, `venv`, etc.) to manage dependencies.* 

## Usage

### Training an SSL Model on a Poisoned Dataset

To train an SSL model (e.g., using MoCo v2) with a chosen poisoning attack, you can use the provided scripts. Example for Distributed Data Parallel (DDP) training:

```bash
# Configure your desired attack, SSL method, dataset, etc. in the relevant config file
# (e.g., configs/ssl/moco_config.yaml, configs/poisoning/...)

bash tools/train.sh <path_to_your_config.yaml>
```

*Please refer to the `configs` directory and specific training scripts for detailed usage and parameter options.*

## Performance Benchmarks (Legacy)

*(Note: These results are based on the original implementation before the current refactoring.)*

| Algorithm       | Method | Clean Acc ↑ | Backdoor Acc ↓ | ASR ↑ |
|-----------------|--------|-------------|----------------|-------|
| SSLBKD          | BYOL   | 66.38%       | 23.82%          | 70.2% |
| SSLBKD          | SimCLR | 70.9%       | 49.1%          | 33.9% |
| SSLBKD          | MoCo   | 66.28%       | 33.24%          | 57.6% |
| SSLBKD          | SimSiam| 64.48%       | 29.3%          | 62.2% |
| CorruptEncoder  | BYOL   |     65.48%   |       25.3%      |  9.66%     |
| CorruptEncoder  | SimCLR |       70.14%      |  45.38%  |   36.9%    |
| CorruptEncoder  | MoCo   |   67.04%   |     38.64%           |  37.3%     |
| CorruptEncoder  | SimSiam|     57.54%        |   14.14%   |   79.48%    |

| Algorithm       | Method | Clean Acc ↑ | Backdoor Acc ↓ | ASR ↑ |
|-----------------|--------|-------------|----------------|-------|
| CTRL            | BYOL   | 75.02%       | 30.87%          | 66.95% |
| CTRL            | SimCLR | 70.32%       | 20.82%          | 81.97% |
| CTRL            | MoCo   | 71.01%       | 54.5%          | 34.34% |
| CTRL            | SimSiam| 71.04%       | 50.36%          | 41.43% |

* Data calculated using the 10% available data evaluation protocol from the SSLBKD paper on the lorikeet class of ImageNet-100 and the airplane class of CIFAR-10, respectively.