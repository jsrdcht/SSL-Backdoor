# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/SSL-Backdoor/results/test/weight_decay/moco_weight_decay_0.05
CONFIG=/workspace/SSL-Backdoor/configs/poisoning/trigger_based/na.yaml
ATTACK_ALGORITHM=bp

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"

CUDA_VISIBLE_DEVICES=4,6 python moco/main_moco.py \
                        --dist-url tcp://localhost:10001 \
                        --config ${CONFIG} \
                        --method moco --amp \
                        -a resnet18 --workers 6 \
                        --weight-decay 0.0005 \
                        --attack_algorithm ${ATTACK_ALGORITHM} \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --epochs 300 --save-freq 30 \
                        --save-folder-root ${SAVE_FOLDER_ROOT} \