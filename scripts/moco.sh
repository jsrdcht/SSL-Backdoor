# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/sync/SSL-Backdoor/results/test
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/sslbkd.yaml
ATTACK_ALGORITHM=sslbkd

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"
cd moco

CUDA_VISIBLE_DEVICES=5,7 python main_moco.py \
                        --dist-url tcp://localhost:10001 \
                        --config ${CONFIG} \
                        -a resnet18 --workers 6 \
                        --attack_algorithm ${ATTACK_ALGORITHM} \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --epochs 300 --save-freq 30 \
                        --moco-contr-tau 0.2 \
                        --save-folder-root ${SAVE_FOLDER_ROOT} \