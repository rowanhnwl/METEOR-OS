#!/usr/bin/env python3
"""Evaluate the five-model METEOR ensemble for CACO2 and HCT116."""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / (
    "infer/out/"
    "inhouse_lnp_booster_data_full_withLANCE_20240907.json_infer_results/"
    "mol.infer_results"
)
DEFAULT_MODELS_DIR = PROJECT_ROOT / "weights/deployed"
DEFAULT_LABELS_JSON = (
    PROJECT_ROOT
    / "data/data_json/inhouse_lnp_booster_data_full_withLANCE_20240907.json"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "ensemble_results.json"

LABELS = (
    "in_house_lnp_CACO2_luc",
    "in_house_lnp_HCT116_luc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute raw-mean and z-score-mean five-model ensemble metrics "
            "for CACO2 and HCT116."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing one child directory per inference model",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing one child directory per deployed model",
    )
    parser.add_argument(
        "--model-filename",
        default="weights.pt",
        help="Model file expected inside each deployed-model directory",
    )
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=DEFAULT_LABELS_JSON,
        help="Source JSON used to identify samples with observed labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path",
    )
    parser.add_argument(
        "--expected-models",
        type=int,
        default=5,
        help="Fail unless this many inference result files are found; use 0 to disable",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_inference_pickle(path: Path) -> dict[str, Any]:
    """Load pickle-wrapped CUDA tensors onto CPU."""

    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda value: torch.load(  # type: ignore[attr-defined]
        io.BytesIO(value), map_location="cpu"
    )
    try:
        with path.open("rb") as handle:
            result = pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = original_loader

    if not isinstance(result, dict):
        raise ValueError(f"Expected a dictionary in {path}, got {type(result).__name__}")
    return result


def discover_result_files(results_dir: Path) -> list[tuple[str, Path]]:
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    result_files: list[tuple[str, Path]] = []
    for model_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        candidates = sorted(model_dir.glob("*.out.pkl"))
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one .out.pkl file in {model_dir}, found {len(candidates)}"
            )
        result_files.append((model_dir.name, candidates[0].resolve()))

    if not result_files:
        raise FileNotFoundError(f"No <model>/*.out.pkl files found under {results_dir}")
    return result_files


def as_float_array(value: Any, *, key: str, model: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Non-finite values in {key!r} for model {model}")
    return array


def align_array(
    array: np.ndarray,
    model_ids: list[str],
    canonical_ids: list[str],
    *,
    key: str,
    model: str,
) -> np.ndarray:
    if len(array) != len(model_ids):
        raise ValueError(
            f"Length mismatch for {key!r} in {model}: "
            f"{len(array)} values vs {len(model_ids)} IDs"
        )
    if model_ids == canonical_ids:
        return array
    position = {sample_id: index for index, sample_id in enumerate(model_ids)}
    return array[np.asarray([position[sample_id] for sample_id in canonical_ids])]


def observed_targets(
    source_data: dict[str, Any], canonical_ids: list[str]
) -> dict[str, np.ndarray]:
    targets = {
        label: np.full(len(canonical_ids), np.nan, dtype=np.float64) for label in LABELS
    }
    for index, sample_id in enumerate(canonical_ids):
        if sample_id not in source_data:
            raise KeyError(f"Inference ID {sample_id!r} is absent from the source JSON")
        sample = source_data[sample_id]
        sample_labels = sample.get("labels", {}) if isinstance(sample, dict) else {}
        for label in LABELS:
            value = sample_labels.get(label)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Non-numeric value for {label!r} on sample {sample_id!r}: {value!r}"
                ) from error
            if math.isfinite(numeric_value):
                targets[label][index] = numeric_value
    return targets


def top50_overlap(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the fraction of the target top half recovered in the predicted top half."""

    target_top = targets > np.median(targets)
    prediction_top = predictions > np.median(predictions)
    denominator = int(target_top.sum())
    if denominator == 0:
        raise ValueError("Cannot compute top-50% overlap for a constant target")
    return float(np.logical_and(target_top, prediction_top).sum() / denominator)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(targets, predictions)[0]),
        "pearson": float(pearsonr(targets, predictions)[0]),
        "top50_overlap": top50_overlap(predictions, targets),
    }


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    models_dir = args.models_dir.resolve()
    labels_json = args.labels_json.resolve()
    output_path = args.output.resolve()

    result_files = discover_result_files(results_dir)
    if args.expected_models and len(result_files) != args.expected_models:
        raise ValueError(
            f"Expected {args.expected_models} models, found {len(result_files)}: "
            + ", ".join(model for model, _ in result_files)
        )

    model_names = [model for model, _ in result_files]
    model_paths = [
        (models_dir / model / args.model_filename).resolve() for model in model_names
    ]
    missing_model_paths = [path for path in model_paths if not path.is_file()]
    if missing_model_paths:
        raise FileNotFoundError(
            "Missing deployed model file(s): "
            + ", ".join(str(path) for path in missing_model_paths)
        )

    source_data = load_json(labels_json)
    model_results = [(model, load_inference_pickle(path)) for model, path in result_files]

    first_ids = model_results[0][1].get("lnp_ids")
    if first_ids is None:
        raise KeyError(f"Missing 'lnp_ids' in {result_files[0][1]}")
    canonical_ids = [str(sample_id) for sample_id in first_ids]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError(f"Duplicate lnp_ids in model {model_names[0]}")

    targets = observed_targets(source_data, canonical_ids)
    predictions: dict[str, dict[str, np.ndarray]] = {label: {} for label in LABELS}
    canonical_set = set(canonical_ids)

    for model, result in model_results:
        if "lnp_ids" not in result:
            raise KeyError(f"Missing 'lnp_ids' for model {model}")
        model_ids = [str(sample_id) for sample_id in result["lnp_ids"]]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"Duplicate lnp_ids in model {model}")
        if set(model_ids) != canonical_set:
            missing = sorted(canonical_set - set(model_ids))[:5]
            extra = sorted(set(model_ids) - canonical_set)[:5]
            raise ValueError(
                f"Model {model} has a different ID set; missing={missing}, extra={extra}"
            )

        for label in LABELS:
            prediction_key = f"{label}infer_predict"
            target_key = f"{label}infer_target"
            if prediction_key not in result or target_key not in result:
                raise KeyError(
                    f"Model {model} lacks {prediction_key!r} or {target_key!r}"
                )

            prediction = align_array(
                as_float_array(result[prediction_key], key=prediction_key, model=model),
                model_ids,
                canonical_ids,
                key=prediction_key,
                model=model,
            )
            stored_target = align_array(
                as_float_array(result[target_key], key=target_key, model=model),
                model_ids,
                canonical_ids,
                key=target_key,
                model=model,
            )
            mask = np.isfinite(targets[label])
            if not np.allclose(stored_target[mask], targets[label][mask], atol=1e-6, rtol=0):
                max_difference = float(
                    np.max(np.abs(stored_target[mask] - targets[label][mask]))
                )
                raise ValueError(
                    f"Stored targets disagree with source JSON for {model}/{label}; "
                    f"maximum difference={max_difference}"
                )
            predictions[label][model] = prediction

    ensemble_results: dict[str, Any] = {}
    for label in LABELS:
        mask = np.isfinite(targets[label])
        observed = targets[label][mask]
        stacked = np.stack([predictions[label][model] for model in model_names])
        raw_ensemble = stacked.mean(axis=0)[mask]

        means = stacked.mean(axis=1, keepdims=True)
        stddevs = stacked.std(axis=1, ddof=0, keepdims=True)
        if np.any(stddevs == 0) or not np.all(np.isfinite(stddevs)):
            raise ValueError(f"Cannot standardize constant predictions for {label}")
        zscore_ensemble = ((stacked - means) / stddevs).mean(axis=0)[mask]

        ensemble_results[label] = {
            "n": int(mask.sum()),
            "raw_mean": compute_metrics(raw_ensemble, observed),
            "zscore_mean": compute_metrics(zscore_ensemble, observed),
        }

    output = {
        "model_paths": [str(path) for path in model_paths],
        "infer_results_file_paths": [str(path) for _, path in result_files],
        "ensembled_results": ensemble_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Loaded {len(model_names)} models and aligned {len(canonical_ids)} samples")
    for label, result in ensemble_results.items():
        print(f"\n{label} (n={result['n']})")
        for method in ("raw_mean", "zscore_mean"):
            metrics = result[method]
            print(
                f"  {method:11} "
                f"Spearman={metrics['spearman']:.6f} "
                f"Pearson={metrics['pearson']:.6f} "
                f"Top50 overlap={metrics['top50_overlap']:.6f}"
            )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
