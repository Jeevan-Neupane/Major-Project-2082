#!/usr/bin/env python3
"""Generate a concise, paper-ready Markdown report from a training CSV/log."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_log_config(log_path: Path) -> dict[str, str]:
    values = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if " |   " not in line or " : " not in line:
            continue
        _, body = line.split(" |   ", 1)
        key, value = body.split(" : ", 1)
        values[key.strip()] = value.strip()
    return values


def fmt(value: str | float, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Markdown experiment report")
    parser.add_argument("--run-id", required=True, help="Run ID used in logs/<run-id>.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "logs" / f"{args.run_id}.csv"
    log_path = PROJECT_ROOT / "logs" / f"{args.run_id}.log"
    if not csv_path.is_file() or not log_path.is_file():
        raise FileNotFoundError("Both logs/<run-id>.csv and logs/<run-id>.log are required")
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError("The run has no completed epochs yet")

    best = max(rows, key=lambda row: float(row["val_dice"]))
    last = rows[-1]
    config = parse_log_config(log_path)
    finished = "Training Summary" in log_path.read_text(encoding="utf-8")
    status = "completed" if finished else "in progress"
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{args.run_id}.md"

    report_path.write_text(
        f"# Experiment report: {args.run_id}\n\n"
        f"Status: **{status}**. This report is generated from the local CSV and log.\n\n"
        "## Protocol\n\n"
        f"- Model: `{config.get('MODEL_TYPE', 'unknown')}`\n"
        f"- Dataset: `{config.get('DATASET_NAME', 'unknown')}`\n"
        f"- Split: `{next((line.split('Split manifest: ', 1)[1] for line in log_path.read_text(encoding='utf-8').splitlines() if 'Split manifest: ' in line), 'not recorded')}`\n"
        f"- Seed: `{config.get('RANDOM_SEED', 'unknown')}`\n"
        f"- Augmentation: `{config.get('AUGMENT_TRAIN', 'unknown')}`\n"
        f"- Optimizer / LR: `{config.get('OPTIMIZER', 'unknown')}` / `{config.get('LEARNING_RATE', 'unknown')}`\n"
        f"- Epochs configured / recorded: `{config.get('NUM_EPOCHS', 'unknown')}` / `{last['epoch']}`\n\n"
        "## Best validation epoch\n\n"
        "| Epoch | Dice | IoU | Precision | Recall | Val. loss |\n"
        "| ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| {best['epoch']} | {fmt(best['val_dice'])} | {fmt(best['val_iou'])} | {fmt(best['val_precision'])} | {fmt(best['val_recall'])} | {fmt(best['val_loss'])} |\n\n"
        "## Latest epoch\n\n"
        "| Epoch | Train loss | Val. loss | Dice | Learning rate |\n"
        "| ---: | ---: | ---: | ---: | ---: |\n"
        f"| {last['epoch']} | {fmt(last['train_loss'])} | {fmt(last['val_loss'])} | {fmt(last['val_dice'])} | {last['lr']} |\n\n"
        "## Reporting note\n\n"
        "These are validation results only. Do not report held-out test results until a final checkpoint is selected without consulting the test split."
        "\n",
        encoding="utf-8",
    )
    print(report_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
