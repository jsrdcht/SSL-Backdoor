#!/bin/bash
# CLIP image-text contrastive training example
set -e

export TORCH_HOME=/workspace/hdd1/pretrained_models/torch
PYTHON=/workspace/conda_envs/torch241_cu118_py310/bin/python
REPO=/workspace/SSL-Backdoor

PYTHONPATH=$REPO $PYTHON $REPO/tools/train_clip.py \
    --config $REPO/configs/clip/clip_vit_b16_cc3m.yaml \
    --device_ids 0,1,2,3
