import torch
from PIL import Image

from ssl_backdoor.attacks.clip_backdoor import poison_generator, zeroshot_eval


class RecordingProcessor:
    def __init__(self):
        self.image_sizes = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.image_sizes.append(images.size)
        return {"pixel_values": torch.zeros(1, 3, 2, 2)}


def test_zero_shot_clean_and_backdoor_share_pre_resize(tmp_path, monkeypatch):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (12, 8)).save(image_path)
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(f"image,label\n{image_path},0\n", encoding="utf-8")
    processor = RecordingProcessor()
    trigger_sizes = []

    def record_trigger(image, *_args, **_kwargs):
        trigger_sizes.append(image.size)
        return image

    monkeypatch.setattr(zeroshot_eval, "apply_static_trigger", record_trigger)
    common = {
        "labels_csv": labels_csv,
        "processor": processor,
        "pre_resize": True,
        "pre_resize_size": [7, 5],
    }
    clean = zeroshot_eval._ZeroShotDataset(**common)
    backdoor = zeroshot_eval._ZeroShotDataset(
        **common,
        trigger_args={"trigger_insert": "patch"},
        trigger_path="unused.png",
    )

    clean[0]
    backdoor[0]

    assert processor.image_sizes == [(7, 5), (7, 5)]
    assert trigger_sizes == [(7, 5)]


def test_poison_generator_resizes_before_trigger(tmp_path, monkeypatch):
    image_path = tmp_path / "source.png"
    Image.new("RGB", (12, 8)).save(image_path)
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("image,caption\nsource.png,a caption\n", encoding="utf-8")
    trigger_sizes = []

    def record_trigger(image, *_args, **_kwargs):
        trigger_sizes.append(image.size)
        return image

    monkeypatch.setattr(poison_generator, "apply_static_trigger", record_trigger)
    poison_generator.generate_poison(
        {
            "train_csv": str(train_csv),
            "output_dir": str(tmp_path / "output"),
            "attack_target": "banana",
            "num_poison": 1,
            "seed": 42,
            "pre_resize": True,
            "pre_resize_size": [7, 5],
            "trigger": {"trigger_insert": "patch", "trigger_size": 2},
        }
    )

    assert trigger_sizes == [(7, 5)]
