import os

HUGGINGFACE_MODEL_PATH = os.environ.get("MODEL_ROOT", "pretrained_models")
DATASET_THAT_NEED_TO_TRANSFORM_ENCODER = ['cifar10', 'cifar100', 'gtsrb']
