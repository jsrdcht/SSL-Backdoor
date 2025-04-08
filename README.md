# SSL-Backdoor

SSL-Backdoor is an academic research library focused on poisoning attacks in self-supervised learning (SSL). The project currently implements two attack algorithms: SSLBKD and CorruptEncoder. This library rewrites the SSLBKD library, providing consistent training code while maintaining consistent hyperparameters (in line with SSLBKD) and training settings, making the training results directly comparable. The key features of this library are:
1. Unified poisining and training framework.
2. Retains the hyperparameters of the default implement, ensuring good comparability.
3. Flexible training implementation: SimCLR uses PyTorch Lightning while other algorithms (BYOL, MoCo, SimSiam) use pure PyTorch.
4. Enhanced configuration system with improved poisoning control.

Future updates will support multimodal contrastive learning models.

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

## Supported Attacks

| Algorithm       | Paper                                      |
|-----------------|--------------------------------------------------|
| SSLBKD          | [Backdoor attacks on self-supervised learning](https://doi.org/10.1109/CVPR52688.2022.01298)    CVPR2022 |
| CTRL           | [An Embarrassingly Simple Backdoor Attack on Self-supervised Learning](https://openaccess.thecvf.com/content/ICCV2023/html/Li_An_Embarrassingly_Simple_Backdoor_Attack_on_Self-supervised_Learning_ICCV_2023_paper.html) CVPR2023 |
| CorruptEncoder  | [Data poisoning based backdoor attacks to contrastive learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html) CVPR2024|
| BLTO (only inference)       | [BACKDOOR CONTRASTIVE LEARNING VIA BI-LEVEL TRIGGER OPTIMIZATION](https://openreview.net/forum?id=oxjeePpgSP) ICLR2024|

## Setup
To set up the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/jsrdcht/SSL-Backdoor.git
    cd SSL-Backdoor
    ```

2. [optional] Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Organization
Take `CIFAR10` as an example, organize the dataset as follows:
1. Store the dataset in `data/CIFAR10/train` and `data/CIFAR10/test` directories.
2. Each dataset should be organized in the `ImageFolder` format.
3. Generate the required dataset configuration filelist under `data/CIFAR10`. An example can be found in `data/CIFAR10/sorted_trainset.txt`. We provide a reference code for generating the dataset configuration file in `scripts/all_data.ipynb`.

For ImageNet-100, follow these extra steps to split the dataset based on the SSLBKD's class list:
  ```bash
  python scripts/create_imagenet_subset.py --subset utils/imagenet100_classes.txt --full_imagenet_path <path> --subset_imagenet_path <path>
  ```

### Configuration File
After organizing the data, you need to modify the config file to specify parameters for a single poisoning experiment. The configuration system has been updated to provide more flexible control over the poisoning process. For example, in `sslbkd.yaml`, you need to set the attack target, poisoning rate, etc.

Example config (`configs/poisoning/trigger_based/sslbkd.yaml`):
```yaml
data: /workspace/SSL-Backdoor/data/ImageNet-100/trainset.txt  # Path to dataset configuration file
dataset: imagenet-100  # Dataset name
save_poisons: True  # Whether to save poisons for persistence

# Poisoning configuration
keep_poison_class: True  # Whether to keep the poison class in the dataset
attack_target_list:
  - 0  # Attack target: int
trigger_path_list:
  - /workspace/SSL-Backdoor/poison-generation/triggers/trigger_14.png  # Trigger path
reference_dataset_file_list:
  - /workspace/SSL-Backdoor/data/ImageNet-100/trainset.txt  # Reference set's dataset configuration file
num_reference_list:
  - 650  # Number of reference samples
num_poison_list:
  - 650  # Number of poisons

# Dataset configuration
finetuning_dataset: /workspace/SSL-Backdoor/data/ImageNet-100/10percent_trainset.txt  # Path to finetuning dataset
downstream_dataset: /workspace/SSL-Backdoor/data/ImageNet-100/valset.txt  # Path to downstream dataset

# Trigger configuration
attack_target_word: n01558993  # Attack class name
trigger_insert: patch  # trigger type
trigger_size: 50  # Trigger size
```

### training a ssl model on poisoned dataset
To train a model using the MoCo v2 method with a specific attack algorithm, run the following command:
```bash
bash scripts/train_moco.sh
```
> Note: Most hyperparameters are hardcoded based on SSLBKD. Modify the script if you need to change any parameters.
For CTRL and adaptive, you must specify the `--no_gaussian` flag to disable the Gaussian noise and use ResNet-CIFAR.

### Evaluating a model using linear probing
To evaluate a model using the linear probing method with a specific attack algorithm, run the following command:
```bash
bash scripts/linear_probe.sh
```

## TODO List
- [ ] implement adaptive attack

