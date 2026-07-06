"""Generate poisoned CSV + poisoned images from clean image-caption CSV.

Flow: sample backdoor subset -> apply trigger to images (apply_static_trigger) and save separately ->
replace caption by target strategy -> merge back clean subset -> write new CSV. Training stage only needs to point data.train_csv to this file.
"""
import os
import random

from PIL import Image

from ssl_backdoor.datasets.attacker.triggers import apply_static_trigger
from .caption_targets import build_target_caption, load_templates
from .utils import (build_trigger_args, read_image_caption_csv, resolve_image_path,
                    resolve_image_root, write_csv)


def _prepare_name(cfg, start, end):
    """Follow CleanCLIP prepare_path_name's readable naming: algorithm/target/size/poison_count."""
    algo = cfg["trigger"].get("attack_algorithm") or cfg["trigger"].get("trigger_insert")
    parts = [start, str(cfg["attack_target"]), str(algo), str(cfg["trigger"].get("trigger_size", "na")),
             str(cfg.get("size_train_data") or "all"), str(cfg["num_poison"])]
    if cfg.get("label_consistent"):
        parts.append("label_consistent")
    return "_".join(parts) + end


def _sample_indices(rows, cfg, caption_key):
    """Return (backdoor_indices, clean_indices). label_consistent only selects from samples containing target word."""
    rng = random.Random(cfg.get("seed", 42))
    num_poison, target = cfg["num_poison"], cfg["attack_target"]
    size = cfg.get("size_train_data")
    indices = list(range(len(rows)))

    if cfg.get("label_consistent"):
        pool = [i for i in indices if target in rows[i][caption_key]]
        rng.shuffle(pool)
        backdoor = pool[:num_poison]
        if len(backdoor) < num_poison:
            raise ValueError(f"label_consistent: only {len(backdoor)} samples containing {target!r}, insufficient for {num_poison}")
        backdoor_set = set(backdoor)
        clean = [i for i in indices if i not in backdoor_set]
    else:
        rng.shuffle(indices)
        backdoor = indices[:num_poison]
        clean = indices[num_poison:]

    if size:
        clean = clean[: max(0, size - len(backdoor))]
    return backdoor, clean


def generate_poison(cfg):
    """Execute poison generation, return generated poisoned CSV path."""
    image_key = cfg.get("image_key", "image")
    caption_key = cfg.get("caption_key", "caption")
    delimiter = cfg.get("delimiter", ",")

    rows, fieldnames = read_image_caption_csv(cfg["train_csv"], image_key, caption_key, delimiter)
    src_root = resolve_image_root(cfg["train_csv"], cfg.get("image_root"))
    out_dir = cfg["output_dir"]
    img_subdir = _prepare_name(cfg, "backdoor_images", "")
    out_img_dir = os.path.join(out_dir, img_subdir)
    os.makedirs(out_img_dir, exist_ok=True)

    backdoor_idx, clean_idx = _sample_indices(rows, cfg, caption_key)

    trigger_args = build_trigger_args(cfg["trigger"])
    trigger_path = cfg["trigger"].get("trigger_path")
    # pre_resize: resize image to this resolution (square) before applying trigger. BadCLIP triggers optimized
    # in 224 space, must be deployed at same scale, otherwise processor scaling will blur patch. Default None keeps original behavior.
    pre_resize = cfg["trigger"].get("pre_resize")
    templates = load_templates(cfg.get("caption_templates"), cfg.get("num_templates"), cfg.get("seed", 42))
    rng = random.Random(cfg.get("seed", 42))
    label_consistent = cfg.get("label_consistent", False)

    poisoned_rows, original_rows, skipped = [], [], 0
    for i in backdoor_idx:
        rel_path = rows[i][image_key]
        try:
            image = Image.open(resolve_image_path(rel_path, src_root)).convert("RGB")
            if pre_resize:
                image = image.resize((pre_resize, pre_resize))
            image = apply_static_trigger(image, trigger_args, trigger=trigger_path)
        except Exception as e:
            skipped += 1
            print(f"[warn] Skipping poisoned image {rel_path}: {e}")
            continue
        out_name = f"{img_subdir}/{i:08d}_{os.path.basename(rel_path)}"
        image.save(os.path.join(out_dir, out_name))

        caption = rows[i][caption_key] if label_consistent else \
            build_target_caption(cfg["attack_target"], templates, rng)
        poisoned_rows.append({image_key: out_name, caption_key: caption, "is_backdoor": 1})
        original_rows.append(dict(rows[i]))

    # Clean row original relative paths based on source CSV directory, while new CSV located in out_dir, so convert to absolute paths;
    # Poisoned images already exist in out_dir, use relative paths. ImageCaptionDataset determines case by case for the mix.
    for i in clean_idx:
        row = {image_key: resolve_image_path(rows[i][image_key], src_root),
               caption_key: rows[i][caption_key], "is_backdoor": 0}
        poisoned_rows.append(row)

    out_fields = [image_key, caption_key, "is_backdoor"]
    csv_path = os.path.join(out_dir, _prepare_name(cfg, "backdoor", ".csv"))
    write_csv(csv_path, poisoned_rows, out_fields, delimiter)
    write_csv(os.path.join(out_dir, _prepare_name(cfg, "original_backdoor", ".csv")),
              original_rows, fieldnames, delimiter)

    print(f"[poison] Poisoned {len(original_rows)} images (skipped {skipped}), clean {len(clean_idx)} images, "
          f"total {len(poisoned_rows)} entries -> {csv_path}")
    return csv_path
