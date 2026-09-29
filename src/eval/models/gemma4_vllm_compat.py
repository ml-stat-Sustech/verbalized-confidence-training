import json
import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import hf_hub_download
from packaging.version import InvalidVersion, Version
from safetensors import safe_open
from safetensors.torch import save_file


logger = logging.getLogger(__name__)

_AFFECTED_VLLM_RELEASES = {
    (0, 23, 0),
    (0, 24, 0),
    (0, 25, 0),
    (0, 25, 1),
}
_COMPAT_DIR_NAME = "huggingface_vllm"
_COMPAT_WEIGHTS_NAME = "gemma4-vllm-compat.safetensors"
_INDEX_NAME = "model.safetensors.index.json"
_MARKER_NAME = "gemma4-vllm-compat.json"


def _installed_vllm_version() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def _needs_gemma4_compat(vllm_version: str | None) -> bool:
    if vllm_version is None:
        return False
    try:
        release = Version(vllm_version).release[:3]
    except InvalidVersion:
        return False
    return release in _AFFECTED_VLLM_RELEASES


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _single_safetensors_layout(model_dir: Path) -> tuple[Path, dict[str, str]]:
    index_path = model_dir / _INDEX_NAME
    if index_path.is_file():
        weight_map = _load_json(index_path).get("weight_map", {})
        if not weight_map:
            raise ValueError(f"Empty safetensors index: {index_path}")
        return model_dir, weight_map

    weights_path = model_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Gemma4 checkpoint has no model.safetensors: {model_dir}")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        weight_map = {name: weights_path.name for name in handle.keys()}
    return model_dir, weight_map


def _resolve_base_model_dir(base_model_name_or_path: str) -> Path:
    local_path = Path(base_model_name_or_path)
    if local_path.is_dir():
        return local_path.resolve()
    weights_path = hf_hub_download(
        base_model_name_or_path,
        filename="model.safetensors",
        local_files_only=True,
    )
    return Path(weights_path).parent


def _shared_kv_weight_names(model_config: dict, base_weight_names: set[str]) -> set[str]:
    text_config = model_config.get("text_config") or {}
    num_layers = int(text_config.get("num_hidden_layers", 0))
    num_shared_layers = int(text_config.get("num_kv_shared_layers", 0))
    if num_layers <= 0 or num_shared_layers <= 0:
        return set()

    first_shared_layer = num_layers - num_shared_layers
    prefixes = tuple(
        f"model.language_model.layers.{layer}.self_attn."
        for layer in range(first_shared_layer, num_layers)
    )
    shared_weight_suffixes = (
        ".k_norm.weight",
        ".v_norm.weight",
        ".k_proj.weight",
        ".v_proj.weight",
    )
    return {
        name
        for name in base_weight_names
        if name.startswith(prefixes) and name.endswith(shared_weight_suffixes)
    }


def _load_selected_tensors(
    model_dir: Path,
    weight_map: dict[str, str],
    names: set[str],
) -> dict:
    tensors = {}
    files_to_names: dict[str, list[str]] = {}
    for name in sorted(names):
        files_to_names.setdefault(weight_map[name], []).append(name)
    for filename, file_names in files_to_names.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in file_names:
                tensors[name] = handle.get_tensor(name)
    return tensors


def _link_checkpoint_files(source_dir: Path, compat_dir: Path) -> None:
    for source in source_dir.iterdir():
        if not source.is_file() or source.name == _INDEX_NAME:
            continue
        target = compat_dir / source.name
        if target.exists() or target.is_symlink():
            continue
        try:
            target.symlink_to(os.path.relpath(source, compat_dir))
        except FileExistsError:
            pass


def prepare_gemma4_vllm_checkpoint(
    checkpoint_name: str,
    base_model_name_or_path: str,
    *,
    vllm_version: str | None = None,
) -> str:
    installed_version = vllm_version or _installed_vllm_version()
    if not _needs_gemma4_compat(installed_version):
        return checkpoint_name

    checkpoint_dir = Path(checkpoint_name)
    config_path = checkpoint_dir / "config.json"
    if not checkpoint_dir.is_dir() or not config_path.is_file():
        return checkpoint_name

    model_config = _load_json(config_path)
    if model_config.get("model_type") != "gemma4":
        return checkpoint_name

    source_dir, source_weight_map = _single_safetensors_layout(checkpoint_dir)
    base_dir = _resolve_base_model_dir(base_model_name_or_path)
    base_dir, base_weight_map = _single_safetensors_layout(base_dir)
    required_names = _shared_kv_weight_names(model_config, set(base_weight_map))
    missing_names = required_names - set(source_weight_map)
    if not missing_names:
        return checkpoint_name

    compat_dir = checkpoint_dir.parent / _COMPAT_DIR_NAME
    marker_path = compat_dir / _MARKER_NAME
    if marker_path.is_file():
        marker = _load_json(marker_path)
        if (
            marker.get("source_checkpoint") == str(checkpoint_dir.resolve())
            and set(marker.get("added_weights", [])) == missing_names
            and (compat_dir / _COMPAT_WEIGHTS_NAME).is_file()
            and (compat_dir / _INDEX_NAME).is_file()
        ):
            return str(compat_dir)

    compat_dir.mkdir(parents=True, exist_ok=True)
    _link_checkpoint_files(source_dir, compat_dir)

    tensors = _load_selected_tensors(base_dir, base_weight_map, missing_names)
    temporary_weights = compat_dir / f".{_COMPAT_WEIGHTS_NAME}.{os.getpid()}.tmp"
    save_file(tensors, str(temporary_weights), metadata={"format": "pt"})
    os.replace(temporary_weights, compat_dir / _COMPAT_WEIGHTS_NAME)

    compat_weight_map = dict(source_weight_map)
    compat_weight_map.update({name: _COMPAT_WEIGHTS_NAME for name in missing_names})
    total_size = sum(
        (compat_dir / filename).stat().st_size
        for filename in set(compat_weight_map.values())
    )
    index = {"metadata": {"total_size": total_size}, "weight_map": compat_weight_map}
    temporary_index = compat_dir / f".{_INDEX_NAME}.{os.getpid()}.tmp"
    with temporary_index.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_index, compat_dir / _INDEX_NAME)

    marker = {
        "source_checkpoint": str(checkpoint_dir.resolve()),
        "vllm_version": installed_version,
        "upstream_issue": "https://github.com/vllm-project/vllm/issues/44788",
        "added_weights": sorted(missing_names),
    }
    temporary_marker = compat_dir / f".{_MARKER_NAME}.{os.getpid()}.tmp"
    with temporary_marker.open("w", encoding="utf-8") as handle:
        json.dump(marker, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_marker, marker_path)

    logger.warning(
        "vLLM %s requires the Gemma4 KV-sharing checkpoint adapter: %s",
        installed_version,
        compat_dir,
    )
    return str(compat_dir)
