# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/SSL-Backdoor/results/test/badencoder/cifar10_moco_nocrop_symentric_300shadows_cosine_continualtraining
CONFIG=/workspace/SSL-Backdoor/configs/poisoning/trigger_based/badencoder_cifar10.yaml

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"

# --method simsiam --feature-dim 2048 --fix-pred-lr \
CUDA_VISIBLE_DEVICES=0,2 python moco/main_moco_badencoder2.py \
                        --dist-url tcp://localhost:10002 \
                        --config ${CONFIG} \
                        --method moco \
                        --amp \
                        -a resnet18 --workers 6 \
                        --resume /workspace/SSL-Backdoor/results/test/badencoder/cifar10_moco_nocrop_symentric_300shadows_cosine/checkpoint_0269.pth.tar \
                        --lr 0.06 --batch-size 256 --multiprocessing-distributed \
                        --epochs 300 --save-freq 30 \
                        --save-folder-root ${SAVE_FOLDER_ROOT} \