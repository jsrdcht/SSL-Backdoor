# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER=/workspace/sync/SSL-Backdoor/results/ctrl/stl10_airplane/simclr_300epoch_no_gaussian
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/ctrl_stl10.yaml
ATTACK_ALGORITHM=ctrl
METHOD=simclr


# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER"
cp "$SCRIPT_PATH" "$SAVE_FOLDER/"
cp "$CONFIG" "$SAVE_FOLDER/"

CUDA_VISIBLE_DEVICES=1,3 python ssl_pretrain.py \
                        --config ${CONFIG} \
                        -a resnet18 --num_workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} --method ${METHOD} \
                        --no_gaussian \
                        --lr 0.5 --batch_size 128 \
                        --epochs 300 --save_freq 30 --eval_freq 20 \
                        --save_folder ${SAVE_FOLDER} \