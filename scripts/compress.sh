CUDA_VISIBLE_DEVICES=3 python CompRess/train_student.py \
    --teacher_arch resnet18 \
    --gpu 0 \
    --save_freq 130 --compress_memory_size 2048 --epochs 130 \
    --teacher /workspace/sync/SSL-Backdoor/results/sslbkd/trigger_14_targeted_n07831146/mocov2_300epoch/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,300/checkpoint_0299.pth.tar \
    --student_arch resnet18 \
    --checkpoint_path CompRess/results/sslbkd/HTBA_trigger_14_targeted_n07831146/mocov2_300epoch_5percent \
    /workspace/sync/SSL-Backdoor/data/ImageNet-100/5percent_trainset.txt