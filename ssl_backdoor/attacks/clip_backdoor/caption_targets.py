"""Target caption / target class strategy.

Generator and evaluator share the same target definition: generator replaces backdoor sample captions with
「target class」template text, evaluator uses the same classes/templates to determine if prediction hits target class,
avoiding calibration drift.
"""
import random

# Default single template used in CLIP paper / when generating poisoned captions.
DEFAULT_TEMPLATE = lambda s: f"a photo of a {s}."


def load_classes_config(path):
    """Load classes.py (contains literal ``classes`` and ``templates``)."""
    with open(path) as f:
        return eval(f.read())


def load_templates(templates_path=None, num_templates=None, seed=0, fallback_to_default=True):
    """Return caption template list (``lambda s -> str``).

    - If ``templates_path`` is empty and ``fallback_to_default`` is True, use single template
      ``DEFAULT_TEMPLATE``;
    - ``num_templates`` used to randomly sample a subset from classes.py template pool (template count ablation).
    """
    if not templates_path:
        if fallback_to_default:
            return [DEFAULT_TEMPLATE]
        raise ValueError("No template path provided, and default single template fallback disabled")
    templates = load_classes_config(templates_path)["templates"]
    if num_templates and num_templates < len(templates):
        templates = random.Random(seed).sample(templates, num_templates)
    return templates


def build_target_caption(target, templates, rng):
    """Sample one from template pool, generate target class caption."""
    return rng.choice(templates)(target)


def resolve_target_index(target, classes):
    """Find class index matching target word (exact match first, then substring containment)."""
    for i, c in enumerate(classes):
        if c == target:
            return i
    for i, c in enumerate(classes):
        if target in c:
            return i
    raise ValueError(f"Target word {target!r} not in classes list")
