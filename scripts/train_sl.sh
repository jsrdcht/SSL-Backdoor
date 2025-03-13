save_dir="/workspace/SSL-Backdoor/results/test/sl/cifar10_alpha_neg_1.0"
log_file="${save_dir}/training_log.txt"
alpha_neg=1.0
master_port=29515
dataset=cifar10
true_class=0

# Create directory if it doesn't exist
mkdir -p "${save_dir}"

CUDA_VISIBLE_DEVICES=3,4 python train_sl_model.py \
    --config /workspace/SSL-Backdoor/configs/poisoning/trigger_based/sslbkd_cifar10.yaml \
    --alpha_neg ${alpha_neg} \
    --dataset ${dataset} --true_class ${true_class} \
    --master_port ${master_port} \
    --save_folder "${save_dir}" 2>&1 | tee "${log_file}"