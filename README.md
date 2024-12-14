# SSL-Backdoor

| Algorithm       | Method | Clean Acc ↑ | Backdoor Acc ↓ | ASR ↑ |
|-----------------|--------|-------------|----------------|-------|
| SSLBKD          | BYOL   | 66.38%       | 23.82%          | 70.2% |
| SSLBKD          | SimCLR | 70.9%       | 49.1%          | 33.9% |
| SSLBKD          | MoCo   | 66.28%       | 33.24%          | 57.6% |
| SSLBKD          | SimSiam| 64.48%       | 29.3%          | 62.2% |
| CorruptEncoder  | BYOL   |             |                |       |
| CorruptEncoder  | SimCLR |             |                |       |
| CorruptEncoder  | MoCo   |             |                |       |
| CorruptEncoder  | SimSiam|             |                |       |

* Data calculated using the 10% available data evaluation protocol from the SSLBKD paper on the lorikeet class of ImageNet-100.

## Project Description
SSL-Backdoor is an academic research library focused on poisoning attacks in self-supervised learning (SSL). The project currently implements two attack algorithms: SSLBKD and CorruptEncoder. This library rewrites the SSLBKD library, providing consistent training code while maintaining consistent hyperparameters (in line with SSLBKD) and training settings, making the training results directly comparable. The key features of this library are:
1. Simplified training code implemented by ssl-lightning.
2. Retains the hyperparameters of the respective algorithms, ensuring good algorithm comparability.

Future updates will support multimodal contrastive learning models.

## Supported Attacks

| Algorithm       | Paper                                      |
|-----------------|--------------------------------------------------|
| SSLBKD          | [Backdoor attacks on self-supervised learning](https://doi.org/10.1109/CVPR52688.2022.01298)    CVPR2022 |
| CorruptEncoder  | [Data poisoning based backdoor attacks to contrastive learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Data_Poisoning_based_Backdoor_Attacks_to_Contrastive_Learning_CVPR_2024_paper.html) CVPR2024|

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
3. Generate the required dataset configuration file under `data/CIFAR10`. An example can be found in `data/CIFAR10/sorted_trainset.txt`. We provide a reference code for generating the dataset configuration file in `scripts/all_data.ipynb`.

For ImageNet100, follow these extra steps:
1. Extract the corresponding classes based on the previous work's split.
2. Use the following script to create the subset:
    ```bash
    python scripts/create_imagenet_subset.py --subset utils/imagenet100_classes.txt --full_imagenet_path <path> --subset_imagenet_path <path>
    ```

### Configuration File
After organizing the data, you need to modify the configuration file to specify parameters for a single pre-training poisoning experiment. For example, in `sslbkd.yaml`, you need to set the attack target, poisoning rate, etc.

Example configuration (`configs/poisoning/trigger_based/sslbkd.yaml`):
```yaml
data: {path to dataset configuration file}
dataset: {dataset name}
save_poisons: {whether to save poisons for persistence, if no then the poisons will be save to data/tmp}
if_target_from_other_dataset: {whether the reference set comes from another dataset}
target_other_dataset_configuration_path: {reference set's dataset configuration file}

attack_target_list:
  - {attack target: int}
attack_target_word: {attack class name}
poison_injection_rate: {poisoning rate}
trigger_path: {trigger path}
blend: {whether to mix the trigger at the image level}
trigger_size: {trigger size}
```

### training a ssl model on poisoned dataset
To train a model using the BYOL method with a specific attack algorithm, run the following command:
```bash
bash scripts/train_byol.sh
```
or
```bash
bash scripts/train_ssl.sh
```

## TODO List
1. Attack acc
2. Implement CTRL

