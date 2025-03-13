# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER=/workspace/SSL-Backdoor/results/backog/imagenet100/trigger_14_targeted_0/simclr_300epoch
CONFIG=/workspace/SSL-Backdoor/configs/poisoning/trigger_based/bp.yaml
ATTACK_ALGORITHM=bp
METHOD=simclr


# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER"
cp "$SCRIPT_PATH" "$SAVE_FOLDER/"
cp "$CONFIG" "$SAVE_FOLDER/"

CUDA_VISIBLE_DEVICES=0,2 python ssl_pretrain.py \
                        --config ${CONFIG} \
                        -a resnet18 --num_workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} --method ${METHOD} \
                        --lr 0.5 --batch_size 256 \
                        --epochs 300 --save_freq 30 --eval_freq 20 \
                        --save_folder ${SAVE_FOLDER}
                        # --no_gaussian \