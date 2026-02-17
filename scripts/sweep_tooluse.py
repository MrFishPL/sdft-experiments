import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ToolUse paper sweep.")
    parser.add_argument("--output_root", type=str, default=None, help="Root directory for sweep runs")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--eval_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--warmup_steps", type=int, default=10)
    return parser.parse_args()


def build_sweep_grid() -> list[dict[str, Any]]:
    learning_rates = [5e-6, 1e-5, 5e-5]
    num_train_epochs = [1, 2]
    num_prompts_per_batch = [16, 32, 64]
    ref_model_mixup_alpha = [0.01, 0.02, 0.05]

    grid: list[dict[str, Any]] = []
    for lr, epochs, batch_size, alpha in product(
        learning_rates,
        num_train_epochs,
        num_prompts_per_batch,
        ref_model_mixup_alpha,
    ):
        grid.append(
            {
                "learning_rate": lr,
                "num_train_epochs": epochs,
                "num_prompts_per_batch": batch_size,
                "ref_model_mixup_alpha": alpha,
            }
        )
    return grid


def run_one_config(config: dict[str, Any], args: argparse.Namespace, run_index: int, run_dir: Path) -> dict[str, Any]:
    run_name = (
        f"tooluse_lr{config['learning_rate']}_ep{config['num_train_epochs']}"
        f"_bs{config['num_prompts_per_batch']}_alpha{config['ref_model_mixup_alpha']}_s{args.seed}"
    )
    log_path = run_dir / "train.log"

    command = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.train",
        "--output_dir",
        str(run_dir),
        "--model_name",
        args.model_name,
        "--seed",
        str(args.seed),
        "--learning_rate",
        str(config["learning_rate"]),
        "--num_train_epochs",
        str(config["num_train_epochs"]),
        "--num_prompts_per_batch",
        str(config["num_prompts_per_batch"]),
        "--ref_model_mixup_alpha",
        str(config["ref_model_mixup_alpha"]),
        "--eval_steps",
        str(args.eval_steps),
        "--eval_strategy",
        args.eval_strategy,
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--save_steps",
        str(args.save_steps),
        "--max_prompt_length",
        str(args.max_prompt_length),
        "--max_completion_length",
        str(args.max_completion_length),
        "--warmup_steps",
        str(args.warmup_steps),
        "--run_name",
        run_name,
        "--paper_hparams",
    ]

    with log_path.open("w", encoding="utf-8") as log_file:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, check=False, env=env)

    result: dict[str, Any] = {
        "run_index": run_index,
        **config,
        "seed": args.seed,
        "run_name": run_name,
        "output_dir": str(run_dir),
        "log_path": str(log_path),
        "return_code": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "eval_tooluse_strict_match": None,
        "eval_loss": None,
        "best_model_checkpoint": None,
    }

    trainer_state = run_dir / "trainer_state.json"
    if trainer_state.exists():
        state = json.loads(trainer_state.read_text(encoding="utf-8"))
        result.update(extract_best_eval_metrics(state))

    return result


def extract_best_eval_metrics(trainer_state: dict[str, Any]) -> dict[str, Any]:
    log_history = trainer_state.get("log_history", [])
    eval_rows = [row for row in log_history if "eval_tooluse_strict_match" in row]

    best_metric = None
    best_loss = None
    if eval_rows:
        best = sorted(
            eval_rows,
            key=lambda row: (
                -float(row.get("eval_tooluse_strict_match", float("-inf"))),
                float(row.get("eval_loss", float("inf"))),
            ),
        )[0]
        best_metric = float(best.get("eval_tooluse_strict_match"))
        best_loss = float(best.get("eval_loss", float("inf")))

    return {
        "eval_tooluse_strict_match": best_metric,
        "eval_loss": best_loss,
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
    }


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [row for row in results if row.get("status") == "ok" and row.get("eval_tooluse_strict_match") is not None]
    failed = [row for row in results if row not in successful]

    successful.sort(
        key=lambda row: (
            -float(row["eval_tooluse_strict_match"]),
            float(row.get("eval_loss", float("inf"))),
        )
    )
    return successful + failed


def write_summary(results: list[dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "sweep_summary.json"
    csv_path = output_root / "sweep_summary.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fieldnames = [
        "run_index",
        "status",
        "return_code",
        "learning_rate",
        "num_train_epochs",
        "num_prompts_per_batch",
        "ref_model_mixup_alpha",
        "seed",
        "eval_tooluse_strict_match",
        "eval_loss",
        "best_model_checkpoint",
        "output_dir",
        "log_path",
        "run_name",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or f"runs/tooluse_paper_sweep_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    configs = build_sweep_grid()
    all_results: list[dict[str, Any]] = []
    total = len(configs)

    for index, config in enumerate(configs, start=1):
        run_dir = output_root / f"run_{index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{total}] starting {config}", flush=True)
        result = run_one_config(config=config, args=args, run_index=index, run_dir=run_dir)
        all_results.append(result)
        print(
            f"[{index}/{total}] finished status={result['status']} metric={result.get('eval_tooluse_strict_match')}",
            flush=True,
        )

    ranked = sort_results(all_results)
    write_summary(ranked, output_root)

    top = ranked[0] if ranked else None
    print(f"Sweep complete: {len(ranked)} runs")
    print(f"Summary JSON: {output_root / 'sweep_summary.json'}")
    print(f"Summary CSV: {output_root / 'sweep_summary.csv'}")
    if top and top.get("eval_tooluse_strict_match") is not None:
        print("Best run:")
        print(json.dumps(top, indent=2))


if __name__ == "__main__":
    main()
