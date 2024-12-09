# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER=/workspace/sync/SSL-Backdoor/results/BDB-R/moco/target21_background-imagenet-a_200references/byol_300epoch
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/randombackground.yaml
ATTACK_ALGORITHM=randombackground

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER"
cp "$SCRIPT_PATH" "$SAVE_FOLDER/"
cp "$CONFIG" "$SAVE_FOLDER/"
cd byol

CUDA_VISIBLE_DEVICES=7 python -m train \
                --method byol \
                --config ${CONFIG} --attack_algorithm ${ATTACK_ALGORITHM} \
                --bs 256 --lr 2e-3 --epoch 300 \
                --arch resnet18 --emb 128 \
                --save-freq 30 --eval_every 30 \
                --save_folder ${SAVE_FOLDER} 2>&1 | tee "${SAVE_FOLDER}/log.txt"