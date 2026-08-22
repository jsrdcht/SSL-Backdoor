"""Default tunable parameters for static trigger injection.

Each entry lists the parameters that are meaningful for a trigger family.
Callers may override any field through args; missing fields are filled from
this template before injection.
"""

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
        "alpha": None,
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
        "lindct": False,
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


TRIGGER_ALIASES = {
    "sslbkd": "patch",
    "badclip": "patch",
    "badnet": "patch",
    "corruptencoder": "patch",
    "refool_ghost": "refool",
    "refool_smooth": "refool",
    "refool_blur": "refool",
}

PARAM_ALIASES = {
    "ghost_rate": ("refool_ghost_rate",),
    "offset": ("refool_offset",),
    "sigma": ("refool_sigma",),
    "ghost_alpha": ("refool_ghost_alpha",),
    "sig_delta": ("sig_amplitude",),
    "sig_frequency": ("sig_freq",),
    "wanet_k": ("k",),
    "wanet_strength": ("wanet_s", "s"),
    "wanet_seed": ("seed",),
}


def canonical_trigger_name(trigger_name):
    return TRIGGER_ALIASES.get(trigger_name, trigger_name)


def trigger_defaults(trigger_name):
    return TRIGGER_PARAM_TEMPLATES.get(canonical_trigger_name(trigger_name), {})
