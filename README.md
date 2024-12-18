# SSL-Backdoor

SSL-Backdoor is an academic research library focused on poisoning attacks in self-supervised learning (SSL). The project currently implements two attack algorithms: SSLBKD and CorruptEncoder. This library rewrites the SSLBKD library, providing consistent training code while maintaining consistent hyperparameters (in line with SSLBKD) and training settings, making the training results directly comparable. The key features of this library are:
1. Simplified training code implemented by ssl-lightning.
2. Retains the hyperparameters of the respective algorithms, ensuring good algorithm comparability.

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

* Data calculated using the 10% available data evaluation protocol from the SSLBKD paper on the lorikeet class of ImageNet-100.



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

For ImageNet-100, follow these extra steps:
1. Extract the corresponding classes based on the previous work's split.
2. Use the following script to create the subset:
    ```bash
    python scripts/create_imagenet_subset.py --subset utils/imagenet100_classes.txt --full_imagenet_path <path> --subset_imagenet_path <path>
    ```

### Configuration File
After organizing the data, you need to modify the configuration file to specify parameters for a single pre-training poisoning experiment. For example, in `sslbkd.yaml`, you need to set the attack target, poisoning rate, etc.

Regardless of whether the pre-training data and attack data come from the same configuration file, you need to specify the `reference_dataset_file_list` parameter. For CorruptEncoder attacks, a configuration file for the reference data is at `SSL-Backdoor/poison-generation/poisonencoder_utils/data_config.txt`.

Example configuration (`configs/poisoning/trigger_based/sslbkd.yaml`):
```yaml
data: /workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_trainset.txt  # Path to dataset configuration file
dataset: imagenet-100  # Dataset name
save_poisons: True  # Whether to save poisons for persistence, the default path is /poisons appended to the save_folder
save_poisons_path: # Path to save poisons
if_target_from_other_dataset: False  # Whether the reference set comes from another dataset, always true for corruptencoder 

# Following parameters are one-to-one correspondence
attack_target_list:
  - 6  # Attack target: int
trigger_path_list:
  - /workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_14.png  # Trigger path
reference_dataset_file_list:
  - /workspace/sync/SSL-Backdoor/data/ImageNet-100/ImageNet100_trainset.txt  # Reference set's dataset configuration file
num_poisons_list:
  - 650  # Number of poisons

attack_target_word: n01558993  # Attack class name
trigger_insert: patch  # trigger type
trigger_size: 50  # Trigger size
```



### training a ssl model on poisoned dataset
To train a model using the BYOL method with a specific attack algorithm, run the following command:
```bash
bash scripts/train_ssl.sh
```
> Note: Most hyperparameters are hardcoded based on SSLBKD. Modify the script if you need to change any parameters.

### Evaluating a model using linear probing
To evaluate a model using the linear probing method with a specific attack algorithm, run the following command:
```bash
bash scripts/linear_probe.sh
```

## TODO List
1. make CTRL correctness
2. implement adaptive attack

