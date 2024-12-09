FOLDER_PATH="/workspace/sync/SSL-Backdoor/results/corruptencoder/trigger_16_targeted_n02106550_2/byol_300epoch"

cd byol
for fname in $FOLDER_PATH/*.pth; do
    CUDA_VISIBLE_DEVICES=0 python -m test --dataset imagenet \
                            --train_clean_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/10percent_trainset.txt \
                            --val_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/valset.txt \
                            --attack_algorithm backog --attack_target 26 --attack_target_word n02106550 \
                            --poison_injection_rate 1 \
                            --trigger_path /workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_16.png --trigger_size 60 \
                            --emb 128 --method byol --arch resnet18 \
                            --fname $fname
    CUDA_VISIBLE_DEVICES=0 python -m test --dataset imagenet \
                            --train_clean_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/1percent_trainset.txt \
                            --val_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/valset.txt \
                            --attack_algorithm backog --attack_target 26 --attack_target_word n02106550 \
                            --poison_injection_rate 1 \
                            --trigger_path /workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_16.png --trigger_size 60 \
                            --emb 128 --method byol --arch resnet18 \
                            --fname $fname
done