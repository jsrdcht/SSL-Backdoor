# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/sync/SSL-Backdoor/results/test/simsiam_300epoch_clean
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/sslbkd_cifar10.yaml
ATTACK_ALGORITHM=clean
METHOD=simsiam

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"

CUDA_VISIBLE_DEVICES=4,6 python ssl_pretrain.py \
                        --config ${CONFIG} \
                        -a resnet18 --num_workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} --method ${METHOD} \
                        --lr 5e-2 --batch_size 128 \
                        --epochs 300 --save_freq 30 \
                        --save_folder_root ${SAVE_FOLDER_ROOT} \