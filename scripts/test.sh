FOLDER_PATH="/workspace/sync/SSL-Backdoor/results/corruptencoder/trigger_14_targeted_n01820546/byol_300epoch"

cd byol
for fname in $FOLDER_PATH/*.pth; do
    CUDA_VISIBLE_DEVICES=5 python -m test --dataset imagenet \
                            --train_clean_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/10percent_trainset.txt \
                            --val_file_path /workspace/sync/SSL-Backdoor/data/ImageNet-100-B/valset.txt \
                            --attack_algorithm backog --attack_target 6 --attack_target_word n01820546 \
                            --poison_injection_rate 0.005 \
                            --trigger_path /workspace/sync/SSL-Backdoor/poison-generation/triggers/trigger_14.png --trigger_size 60 \
                            --emb 128 --method byol --arch resnet18 \
                            --fname $fname
done