# 获取脚本自身的绝对路径
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"

OUTPUT_DIR=/workspace/sync/SSL-Backdoor/results/pbcl/trigger_14_targeted_n07831146/mae_800epoch_five_trigger
DATA_PATH=/workspace/sync/SSL-Backdoor/poison-generation/data/ImageNet-100/HTBA_trigger_14_targeted_n07831146_five_trigger/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_60_rate_0.50_targeted_True_filelist.txt

# 将脚本内容复制到目标目录
mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/"
cd mae

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=12346 \
    main_pretrain.py \
    --batch_size 128 --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --data_path ${DATA_PATH}
