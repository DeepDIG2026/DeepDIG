"""Strict loading of current and previously published DeepDIG checkpoints."""

from collections import OrderedDict
from collections.abc import Mapping

import torch


# Module member names changed when RATA and CADC became explicit submodules.
_LEGACY_PREFIXES = (
    ("dynamic_path.tadc.accumulator.content_mlp", "dynamic_path.radc.rata.region_reliability_mlp"),
    ("dynamic_path.tadc.accumulator.spatial_gate", "dynamic_path.radc.rata.pixel_discriminability_gate"),
    ("dynamic_path.tadc.accumulator", "dynamic_path.radc.rata"),
    ("dynamic_path.tadc.kernel_generator", "dynamic_path.radc.cadc.kernel_generator"),
    ("dynamic_path.tadc.bias", "dynamic_path.radc.cadc.bias"),
    ("dynamic_path.tadc", "dynamic_path.radc"),
    ("difference_path.ca", "difference_path.channel_attention"),
    ("difference_path.sa", "difference_path.spatial_attention"),
    ("dynamic_path.ca", "dynamic_path.channel_attention"),
    ("dynamic_path.sa", "dynamic_path.spatial_attention"),
    ("mag_module.static_enhance.gate_encoder", "mag_module.static_gating_generator.encoder"),
    ("mag_module.dynamic_enhance.gate_encoder", "mag_module.dynamic_gating_generator.encoder"),
    ("mag_module.fusion.ca", "mag_module.fusion.channel_attention"),
    ("mag_module.fusion.sa", "mag_module.fusion.spatial_attention"),
    ("mag_module.diff_proj", "mag_module.difference_projection"),
    ("sp", "static_path"),
)


def _canonical_name(name):
    if name == "module":
        return ""
    if name.startswith("module."):
        name = name[7:]
    for old, new in _LEGACY_PREFIXES:
        if name == old or name.startswith(old + "."):
            return new + name[len(old):]
    return name


def load_deepdig_checkpoint(model, checkpoint):
    """Load a full checkpoint or raw state dictionary, with strict validation.

    Published checkpoints may use legacy module names and duplicate the static
    path under ``sp``. Aliases are merged only when their tensors agree; missing,
    unexpected, or incorrectly shaped weights still fail strict loading.
    Both current names and an optional DataParallel ``module.`` prefix work.
    """
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Expected a DeepDIG checkpoint or state dictionary")
    if checkpoint.get("pipeline_ablation", "full") != "full":
        raise ValueError("A complete DBA + RADC + MAG checkpoint is required")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint state_dict must be a mapping")

    canonical = OrderedDict()
    origins = {}
    for name, tensor in state.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("State dictionary entries must be named tensors")
        target = _canonical_name(name)
        if target in canonical:
            previous = canonical[target]
            if previous.dtype != tensor.dtype or not torch.equal(previous, tensor):
                raise ValueError(
                    f"Conflicting checkpoint aliases: {origins[target]!r} and {name!r}"
                )
        else:
            canonical[target] = tensor
            origins[target] = name

    if hasattr(state, "_metadata"):
        canonical._metadata = OrderedDict()
        for name, metadata in state._metadata.items():
            target = _canonical_name(name)
            if target in canonical._metadata and canonical._metadata[target] != metadata:
                raise ValueError(f"Conflicting checkpoint metadata for {target!r}")
            canonical._metadata[target] = metadata

    return model.load_state_dict(canonical, strict=True)


__all__ = ["load_deepdig_checkpoint"]
