# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/sync/SSL-Backdoor/results/test/simclr_300epoch_ctrl
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/ctrl.yaml
ATTACK_ALGORITHM=ctrl
METHOD=simclr
# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"

CUDA_VISIBLE_DEVICES=2,7 python ssl_pretrain.py \
                        --config ${CONFIG} \
                        -a resnet18 --num_workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} --method ${METHOD} \
                        --lr 0.5 --batch_size 128 \
                        --epochs 300 --save_freq 30 \
                        --save_folder_root ${SAVE_FOLDER_ROOT} \