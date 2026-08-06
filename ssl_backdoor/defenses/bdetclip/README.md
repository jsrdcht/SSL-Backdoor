# BDetCLIP

This module implements BDetCLIP (ICML 2025), a training-free test-time defense for backdoored
CLIP models. It compares image similarity to benign and malignant class prompts. A smaller raw
difference (`omega`) and therefore a larger anomaly score (`-omega`) indicates a more likely
backdoor sample.

## Usage

Install the project environment, update `model.checkpoint` and `data.image_root` in the config,
then run:

```bash
pixi run -e cuda python tools/run_bdetclip.py --config configs/bdetclip/sslbkd.yaml
```

Use `configs/bdetclip/badclip.yaml` for BadCLIP. The portable shell wrapper accepts the config as
its first argument and forwards the remaining options:

```bash
tools/run_bdetclip.sh configs/bdetclip/badclip.yaml --device cuda:0
```

By default, the evaluator deterministically selects 200 clean reference images and 1,000 disjoint
evaluation images from ImageNet validation data, then applies the configured trigger online to 30%
of non-target evaluation samples.

The paper threshold is the minimum raw score over clean reference images. Each run writes the
resolved config, log, aggregate metrics, and per-sample scores. Zero-shot clean accuracy and ASR use
the configurable prediction template, which defaults to `a photo of a {}`.

## Resources and attribution

The benign and malignant prompt resources in `assets/bdetclip` originate from the official
[BDetCLIP implementation](https://github.com/Purshow/BDetCLIP). Follow the upstream repository's
license and non-commercial-use notice when redistributing or adapting those materials. The ImageNet
class definitions come from `assets/imagenet/classes.py` in this repository.

## Reference regression results

These results were measured with epoch-10 poisoned checkpoints, 200 reference images, and 1,000
evaluation images (30% poisoned):

| Attack | Clean accuracy | ASR | AUROC | AUPRC |
|---|---:|---:|---:|---:|
| BadCLIP | 0.4871 | 0.9467 | 0.7473 | 0.4330 |
| SSLBKD | 0.5443 | 0.9900 | 0.7822 | 0.4837 |

The clean-reference threshold did not detect poisoned samples in these two runs (threshold F1 = 0).
AUROC is threshold-independent and is the primary regression metric here.
