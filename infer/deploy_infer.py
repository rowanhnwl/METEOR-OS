#!/usr/bin/env python3
"""Run inference for every deployed model, then ensemble each dataset's results."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
from pathlib import Path

from pyprojroot import here


PROJECT_ROOT = Path(here())
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def actual_path(path: Path) -> Path:
    """Resolve configuration paths relative to the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    config_path = actual_path(path)
    with config_path.open() as handle:
        return json.load(handle)


def discover_child_directories(configured_path: Path) -> list[Path]:
    directory = actual_path(configured_path)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return [
        configured_path / child.name
        for child in sorted(directory.iterdir())
        if child.is_dir()
    ]


def discover_dataset_jobs(
    infer_data_path: Path,
    output_dir: Path,
    output_append_string: str,
    labels_json_dir: Path,
) -> list[dict[str, Path]]:
    jobs: list[dict[str, Path]] = []
    for deploy_data_dir in discover_child_directories(infer_data_path):
        real_data_dir = actual_path(deploy_data_dir)
        for candidate in sorted(real_data_dir.iterdir(), reverse=True):
            if not candidate.name.endswith("lmdb"):
                continue

            dataset_lmdb_path = deploy_data_dir / candidate.name
            infer_output_dir = output_dir / (
                deploy_data_dir.name + output_append_string
            )
            results_dir = infer_output_dir / candidate.name.replace(
                "lmdb", "infer_results"
            )
            jobs.append(
                {
                    "dataset_lmdb_path": dataset_lmdb_path,
                    "results_dir": results_dir,
                    "labels_json": labels_json_dir / deploy_data_dir.name,
                }
            )

    if not jobs:
        raise FileNotFoundError(
            f"No *.lmdb datasets found directly under {actual_path(infer_data_path)}"
        )
    return jobs


def inference_already_done(output_dir: Path) -> bool:
    real_output_dir = actual_path(output_dir)
    if not real_output_dir.is_dir():
        return False
    result_files = sorted(real_output_dir.glob("*.out.pkl"))
    if len(result_files) > 1:
        raise ValueError(
            f"Expected at most one .out.pkl file in {real_output_dir}, "
            f"found {len(result_files)}"
        )
    if result_files:
        print(f"Inference already done: {result_files[0]}", flush=True)
        return True
    return False


def build_inference_command(
    config: dict,
    dataset_lmdb_path: Path,
    model_path: Path,
    results_path: Path,
) -> list[str]:
    task_name = dataset_lmdb_path.parent
    command = [
        sys.executable,
        "unimol/infer_np.py",
        "--user-dir",
        "unimol",
        str(config["data_path"]),
        "--task-name",
        str(task_name),
        "--valid-subset",
        "infer",
        "--num-workers",
        "8",
        "--ddp-backend=c10d",
        "--batch-size",
        str(config["eval_batch_size"]),
        "--task",
        "mol_np_finetune",
        "--loss",
        str(config["loss_func"]),
        "--arch",
        "np_unimol",
        "--classification-head-name",
        str(task_name),
        "--num-classes",
        str(config["task_num"]),
        "--dict-name",
        str(Path("infer/misc") / config["dict_name"]),
        "--conf-size",
        str(config["conf_size"]),
        "--only-polar",
        str(config["only_polar"]),
        "--path",
        str(model_path),
        "--fp16",
        "--fp16-init-scale",
        "4",
        "--fp16-scale-window",
        "256",
        "--log-interval",
        "50",
        "--log-format",
        "simple",
        "--results-path",
        str(results_path),
        "--lnp-encoder-layers",
        str(config["lnp_encoder_layers"]),
        "--lnp-encoder-embed-dim",
        str(config["lnp_encoder_embed_dim"]),
        "--lnp-encoder-ffn-embed-dim",
        str(config["lnp_encoder_ffn_embed_dim"]),
        "--lnp-encoder-attention-heads",
        str(config["lnp_encoder_attention_heads"]),
        "--full-dataset-task-schema-path",
        str(config["schema_path"]),
        "--load-full-np-model",
        "--concat-datasets",
    ]
    if config["include_excipients"]:
        command.append("--include-excipients")
    return command


def run_ensemble(
    config: dict,
    results_dir: Path,
    labels_json: Path,
    expected_models: int,
) -> None:
    ensemble_script = actual_path(
        Path(config.get("ensemble_script", "infer/ensemble_predictions.py"))
    )
    if not ensemble_script.is_file():
        raise FileNotFoundError(f"Ensembling script does not exist: {ensemble_script}")

    real_labels_json = actual_path(labels_json)
    if not real_labels_json.is_file():
        raise FileNotFoundError(
            f"Source-label JSON required for ensembling does not exist: {real_labels_json}"
        )

    output_path = results_dir / config.get(
        "ensemble_output_filename", "ensemble_results.json"
    )
    command = [
        sys.executable,
        str(ensemble_script),
        "--results-dir",
        str(results_dir),
        "--models-dir",
        str(config["deploy_model_master_dir"]),
        "--model-filename",
        str(config["deploy_weight_filename"]),
        "--labels-json",
        str(labels_json),
        "--output",
        str(output_path),
        "--expected-models",
        str(expected_models),
    ]
    print(f"Running ensemble for {results_dir}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft_limit = 8192 if hard_limit == resource.RLIM_INFINITY else min(8192, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))

    infer_data_path = Path(config["infer_data_path"])
    models_dir = Path(config["deploy_model_master_dir"])
    output_dir = Path(config["output_dir"])
    labels_json_dir = Path(config.get("ensemble_labels_json_dir", "data/data_json"))
    run_ensembling = bool(config.get("run_ensemble", True))

    model_dirs = discover_child_directories(models_dir)
    if not model_dirs:
        raise FileNotFoundError(
            f"No deployed model directories found under {actual_path(models_dir)}"
        )

    for model_dir in model_dirs:
        model_path = model_dir / config["deploy_weight_filename"]
        if not actual_path(model_path).is_file():
            raise FileNotFoundError(
                f"Deployed model does not exist: {actual_path(model_path)}"
            )

    dataset_jobs = discover_dataset_jobs(
        infer_data_path=infer_data_path,
        output_dir=output_dir,
        output_append_string=config["output_append_string"],
        labels_json_dir=labels_json_dir,
    )

    if run_ensembling:
        for job in dataset_jobs:
            if not actual_path(job["labels_json"]).is_file():
                raise FileNotFoundError(
                    "Source-label JSON required for ensembling does not exist: "
                    f"{actual_path(job['labels_json'])}"
                )

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_path = model_dir / config["deploy_weight_filename"]
        for job in dataset_jobs:
            model_results_dir = job["results_dir"] / model_name
            actual_path(model_results_dir).mkdir(parents=True, exist_ok=True)
            if inference_already_done(model_results_dir):
                continue

            command = build_inference_command(
                config=config,
                dataset_lmdb_path=job["dataset_lmdb_path"],
                model_path=model_path,
                results_path=model_results_dir,
            )
            print(
                f"Running inference for model {model_name} on "
                f"{job['dataset_lmdb_path']}",
                flush=True,
            )
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    if run_ensembling:
        for job in dataset_jobs:
            run_ensemble(
                config=config,
                results_dir=job["results_dir"],
                labels_json=job["labels_json"],
                expected_models=len(model_dirs),
            )


if __name__ == "__main__":
    main()
