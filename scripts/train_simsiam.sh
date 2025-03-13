# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/SSL-Backdoor/results/test/num_reference/moco_reference_1200
CONFIG=/workspace/SSL-Backdoor/configs/poisoning/trigger_based/na_cifar10.yaml
ATTACK_ALGORITHM=bp

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"

CUDA_VISIBLE_DEVICES=2,6 python moco/main_moco.py \
  --config ${CONFIG} \
  -a resnet18 --amp --feature-dim 2048 --method simsiam \
  --workers 6 \
  --attack_algorithm ${ATTACK_ALGORITHM} \
  --lr 0.1 --batch-size 256 --epochs 300 --save-freq 30 \
  --dist-url 'tcp://localhost:10008' --multiprocessing-distributed \
  --fix-pred-lr \
  --save-folder-root ${SAVE_FOLDER_ROOT}