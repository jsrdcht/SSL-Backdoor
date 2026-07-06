# CLIP Configuration Path Guide

This document explains the path configurations in CLIP config files and how to adapt them to your environment.

## Paths That Need Configuration

### 1. Pretrained Model Path
**Current**: `${MODEL_ROOT}/clip-vit-base-patch16`
**Update to**: Your local CLIP model path or HuggingFace model identifier

```yaml
model:
  path: openai/clip-vit-base-patch16  # HuggingFace model ID (recommended)
  # OR
  path: /your/path/to/clip-vit-base-patch16  # Local path
```

### 2. Dataset Paths

#### Training Data (CC3M)
**Current**: `${DATA_ROOT}/cc3m`
**Update to**: Your CC3M dataset path

```yaml
data:
  train_csv: /your/path/to/cc3m/train.csv
  image_root: /your/path/to/cc3m/images
```

#### ImageNet Validation
**Current**: `${PROJECT_ROOT}/ssl_backdoor/projects/CleanCLIP/data/ImageNet1K/validation/labels.csv`
**Update to**: Your ImageNet validation labels

```yaml
labels_csv: /your/path/to/imagenet/val/labels.csv
```

The `labels.csv` format should be:
```
image,label
ILSVRC2012_val_00000001.JPEG,0
ILSVRC2012_val_00000002.JPEG,1
...
```

### 3. Output Paths

#### Experiment Results
**Current**: `${PROJECT_ROOT}/results/...`
**Update to**: Your desired output directory

```yaml
save_folder_root: ./results  # Relative to repository root
experiment_id: clip_backdoor_experiment
```

#### Poisoned Data Output
**Current**: `${PROJECT_ROOT}/data/clip_backdoor/...`
**Update to**: Your data output directory

```yaml
output_dir: ./data/poisoned  # Relative to repository root
```

### 4. Trigger Assets

#### Existing Triggers
**Current**: `${PROJECT_ROOT}/assets/triggers/trigger_14.png`
**Already Fixed**: `assets/triggers/trigger_14.png` (relative path)

Note: You'll need to add trigger images to `assets/triggers/` directory.

#### BadCLIP Generated Triggers
**Current**: `${PROJECT_ROOT}/results/.../badclip_trigger.png`
**Update to**: Your output path

```yaml
trigger_optimization:
  output_trigger_path: ./results/badclip/trigger.png
```

### 5. BadCLIP Positive Samples
**Current**: `${PROJECT_ROOT}/data/badclip/banana_samples_from_cc3m_existing.csv`
**Update to**: Generated positive samples path

Generate using:
```bash
python tools/build_badclip_positive_samples.py \
  --train_csv /your/path/to/cc3m/train.csv \
  --target_label banana \
  --output_csv ./data/badclip/banana_samples.csv \
  --max_samples 500
```

## Recommended Directory Structure

```
SSL-Backdoor-github/
├── assets/
│   ├── imagenet/
│   │   └── classes.py          # ✅ Already included
│   └── triggers/
│       └── trigger_14.png      # ⚠️ Add your triggers here
├── configs/
│   └── clip/
│       ├── badclip/
│       └── clip_backdoor/
├── data/                        # Create as needed
│   ├── badclip/
│   │   └── banana_samples.csv
│   ├── cc3m/                    # Your CC3M dataset
│   └── imagenet/                # Your ImageNet dataset
├── results/                     # Created automatically
│   ├── badclip/
│   └── clip_backdoor/
└── pretrained_models/           # Optional: local models
    └── clip-vit-base-patch16/
```

## Quick Start Configuration

### Option 1: Use HuggingFace Models (Recommended)
No need to download models manually:
```yaml
model:
  path: openai/clip-vit-base-patch16
```

### Option 2: Use Local Paths
1. Download CLIP model from HuggingFace
2. Update all `path:` fields in configs
3. Set environment variables:
```bash
export CLIP_MODEL_PATH=/your/path/to/clip-vit-base-patch16
export CC3M_PATH=/your/path/to/cc3m
export IMAGENET_PATH=/your/path/to/imagenet
```

## Config Files That Need Updates

- `configs/clip/clip_vit_b16_cc3m.yaml`
- `configs/clip/badclip/optimize_trigger_banana.yaml`
- `configs/clip/badclip/poison_badclip_banana_cc3m.yaml`
- `configs/clip/badclip/train_badclip_banana_cc3m.yaml`
- `configs/clip/badclip/eval_zeroshot_imagenet.yaml`
- `configs/clip/clip_backdoor/clip_vit_b16_cc3m_poisoned.yaml`
- `configs/clip/clip_backdoor/poison_sslbkd_banana_cc3m.yaml`
- `configs/clip/clip_backdoor/eval_zeroshot_imagenet.yaml`

## Next Steps

1. Review each config file in `configs/clip/`
2. Update paths according to your environment
3. Create necessary directories
4. Generate positive samples for BadCLIP if needed
5. Run the attacks!
