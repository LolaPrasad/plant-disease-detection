"""
Shared torchvision transforms for training and evaluation.

All transforms consume the config dict produced by utils.config.load_config()
so hyperparameters live only in YAML, not here.
"""

from __future__ import annotations

from typing import Any

import torchvision.transforms as T


def get_train_transforms(cfg: dict[str, Any]) -> T.Compose:
    aug = cfg["augmentation"]["train"]
    norm = cfg["normalization"]
    input_size: int = cfg.get("model", {}).get("input_size", 224)

    steps: list = []

    # Spatial augmentations
    if "random_resized_crop" in aug:
        steps.append(T.RandomResizedCrop(
            size=input_size,
            scale=tuple(aug["random_resized_crop"]["scale"]),
        ))
    else:
        steps.append(T.Resize(input_size))

    if aug.get("horizontal_flip"):
        steps.append(T.RandomHorizontalFlip())

    if aug.get("vertical_flip"):
        steps.append(T.RandomVerticalFlip())

    if "random_rotation" in aug:
        steps.append(T.RandomRotation(degrees=aug["random_rotation"]))

    if "random_affine" in aug:
        ra = aug["random_affine"]
        steps.append(T.RandomAffine(
            degrees=ra.get("degrees", 0),
            translate=tuple(ra["translate"]) if "translate" in ra else None,
            scale=tuple(ra["scale"]) if "scale" in ra else None,
        ))

    if "random_perspective" in aug:
        rp = aug["random_perspective"]
        steps.append(T.RandomPerspective(
            distortion_scale=rp.get("distortion_scale", 0.2),
            p=rp.get("p", 0.3),
        ))

    # Colour augmentations
    if "color_jitter" in aug:
        cj = aug["color_jitter"]
        steps.append(T.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0),
        ))

    if "gaussian_blur" in aug:
        gb = aug["gaussian_blur"]
        steps.append(T.RandomApply(
            [T.GaussianBlur(kernel_size=gb["kernel_size"])],
            p=gb.get("p", 0.2),
        ))

    steps += [
        T.ToTensor(),
        T.Normalize(mean=norm["mean"], std=norm["std"]),
    ]

    return T.Compose(steps)


def get_val_transforms(cfg: dict[str, Any]) -> T.Compose:
    vt = cfg["augmentation"]["val_test"]
    norm = cfg["normalization"]

    return T.Compose([
        T.Resize(vt["resize"]),
        T.CenterCrop(vt["center_crop"]),
        T.ToTensor(),
        T.Normalize(mean=norm["mean"], std=norm["std"]),
    ])
