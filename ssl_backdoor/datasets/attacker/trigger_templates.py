"""Default tunable parameters for static trigger injection.

Each entry lists the parameters that are meaningful for a trigger family.
Callers may override any field through args; missing fields are filled from
this template before injection.
"""

from collections.abc import Mapping

TRIGGER_PARAM_TEMPLATES = {
    "patch": {
        "trigger_size": 50,
        "position": "random",
        "location_min": 0.25,
        "location_max": 0.75,
        "alpha": 1.0,
    },
    "blend": {
        "alpha": 0.2,
    },
    "refool": {
        "alpha_t": None,
        "ghost_rate": 0.49,
        "refool_max_image_size": None,
        "refool_preserve_size": True,
        "offset": (0, 0),
        "sigma": -1.0,
        "ghost_alpha": -1.0,
    },
    "ctrl": {
        "channel_list": [1, 2],
        "window_size": 32,
        "pos_list": [(15, 15), (31, 31)],
        "attack_magnitude": 50,
    },
    "sig": {
        "sig_delta": 20.0,
        "sig_frequency": 6.0,
        "sig_direction": "horizontal",
    },
    "wanet": {
        "wanet_k": 4,
        "wanet_strength": 2.0,
        "wanet_seed": 0,
    },
    "blto": {
        "generator_path": None,
        "device": "cpu",
    },
}


TRIGGERS_REQUIRING_PATH = frozenset({"patch", "blend", "refool"})
_REMOVED_TRIGGER_NAMES = frozenset(
    {"refool_ghost", "refool_smooth", "refool_blur"}
)
_INTERNAL_TRIGGER_FIELDS = frozenset(
    {"_ctrl_poisoning_agent", "_blto_poisoning_agent", "_wanet_map_cache"}
)


def trigger_defaults(trigger_name):
    try:
        return TRIGGER_PARAM_TEMPLATES[trigger_name]
    except KeyError as exc:
        supported = ", ".join(TRIGGER_PARAM_TEMPLATES)
        raise ValueError(
            f"Unsupported trigger_insert {trigger_name!r}; supported values: {supported}"
        ) from exc


def resolve_trigger_name(attack_algorithm=None, trigger_insert=None):
    """Resolve one canonical trigger without silently overriding conflicts."""
    if attack_algorithm in _REMOVED_TRIGGER_NAMES:
        raise ValueError(f"Unsupported attack_algorithm {attack_algorithm!r}")
    if trigger_insert is None:
        supported = ", ".join(TRIGGER_PARAM_TEMPLATES)
        raise ValueError(
            "trigger_insert must select one of: "
            f"{supported}; got attack_algorithm={attack_algorithm!r}"
        )

    trigger_defaults(trigger_insert)
    if (
        attack_algorithm in TRIGGER_PARAM_TEMPLATES
        and attack_algorithm != trigger_insert
    ):
        raise ValueError(
            "Conflicting trigger selectors: "
            f"attack_algorithm={attack_algorithm!r}, "
            f"trigger_insert={trigger_insert!r}"
        )
    return trigger_insert


def validate_trigger_config(trigger_cfg, require_path=False, allow_internal=False):
    """Validate a dedicated trigger mapping against the canonical schema."""
    if not isinstance(trigger_cfg, Mapping):
        raise TypeError("trigger config must be a mapping")
    if trigger_cfg.get("trigger_insert") is None:
        raise ValueError("trigger config must provide trigger_insert")

    trigger_name = resolve_trigger_name(trigger_insert=trigger_cfg["trigger_insert"])
    allowed = {"trigger_insert", *trigger_defaults(trigger_name)}
    if trigger_name in TRIGGERS_REQUIRING_PATH:
        allowed.add("trigger_path")
    if allow_internal:
        allowed.update(_INTERNAL_TRIGGER_FIELDS)

    unknown = sorted(key for key in trigger_cfg if key not in allowed)
    if unknown:
        raise ValueError(
            f"Unsupported fields for trigger_insert={trigger_name!r}: {unknown}"
        )
    if require_path and trigger_name in TRIGGERS_REQUIRING_PATH:
        if not trigger_cfg.get("trigger_path"):
            raise ValueError(
                f"trigger_path is required for trigger_insert={trigger_name!r}"
            )
    return trigger_name
