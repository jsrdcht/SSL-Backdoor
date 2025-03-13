# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/SSL-Backdoor/results/test/blto/imagenet100-n/byol
CONFIG=/workspace/SSL-Backdoor/configs/poisoning/optimization_based/adaptive_poisoning.yaml
ATTACK_ALGORITHM=blto

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"
cd byol

CUDA_VISIBLE_DEVICES=7 python -m train \
                --method byol \
                --config ${CONFIG} --attack_algorithm ${ATTACK_ALGORITHM} \
                --bs 256 --lr 2e-3 --epoch 300 \
                --arch resnet18 --emb 128 \
                --save-freq 30 --eval_every 30 \
                --save_folder ${SAVE_FOLDER_ROOT} 2>&1 | tee "${SAVE_FOLDER}/log.txt"