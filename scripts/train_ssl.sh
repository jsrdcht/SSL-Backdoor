# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER=/workspace/sync/SSL-Backdoor/results/test/imagenet100/optimized_classalign_multipoisons_BYOL
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/optimization_based/optimized_poisoning.yaml
ATTACK_ALGORITHM=optimized
METHOD=byol

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER"
cp "$SCRIPT_PATH" "$SAVE_FOLDER/"
cp "$CONFIG" "$SAVE_FOLDER/"

CUDA_VISIBLE_DEVICES=5,6 python ssl_pretrain.py \
                        --config ${CONFIG} \
                        -a resnet18 --num_workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} --method ${METHOD} \
                        --no_gaussian \
                        --lr 2e-3 --batch_size 256 \
                        --epochs 300 --save_freq 30 --eval_freq 30 \
                        --save_folder ${SAVE_FOLDER} \