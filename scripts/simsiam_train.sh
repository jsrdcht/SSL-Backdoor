# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

# 指定保存目录，也可以是其他目录变量
SAVE_FOLDER_ROOT=/workspace/sync/SSL-Backdoor/results/different_downstream/trigger_14_pretrain_imagenet100_downstream_stl10_0/simsiam_300epoch
CONFIG=/workspace/sync/SSL-Backdoor/configs/poisoning/trigger_based/backog.yaml
ATTACK_ALGORITHM=backog

# 将脚本内容复制到目标目录
mkdir -p "$SAVE_FOLDER_ROOT"
cp "$SCRIPT_PATH" "$SAVE_FOLDER_ROOT/"
cp "$CONFIG" "$SAVE_FOLDER_ROOT/"
cd simsiam

CUDA_VISIBLE_DEVICES=2,3 python main_simsiam.py \
  --config ${CONFIG} \
  -a resnet18 --save-freq 60 \
  --attack_algorithm ${ATTACK_ALGORITHM} \
  --lr 0.05 --batch-size 256 --epochs 300 \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  --fix-pred-lr \
  --save-folder ${SAVE_FOLDER_ROOT}