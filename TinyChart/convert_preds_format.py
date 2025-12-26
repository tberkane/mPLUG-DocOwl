#!/usr/bin/env python3
"""
Create a new predict output directory compatible with vlm-ensemble-chart-extraction.

Example:
python make_pred_dir.py \
  --input_json /lab-share/CHIP-Majumder-PNR-e2/Public/thomas/mPLUG-DocOwl/TinyChart/outputs/chartqa_preds.json \
  --out_root /lab-share/CHIP-Majumder-PNR-e2/Public/thomas/vlm-ensemble-chart-extraction/outputs/predict \
  --input_images_dir "data/ChartQA Dataset/test/png" \
  --model TinyChart \
  --temperature 0.0
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_tag() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=False)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON on line {ln}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Line {ln} is not a JSON object.")
            rows.append(obj)
    return rows


def _write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_yaml_minimal(path: Path, d: Dict[str, Any]) -> None:
    # Minimal YAML writer to avoid extra deps. Good enough for simple flat configs.
    lines = []
    for k, v in d.items():
        if isinstance(v, str):
            # Quote strings that contain special chars/spaces
            if any(ch in v for ch in [":", "#", "{", "}", "[", "]", "\n", "\t"]) or " " in v:
                vv = '"' + v.replace('"', '\\"') + '"'
            else:
                vv = v
        elif isinstance(v, bool):
            vv = "true" if v else "false"
        else:
            vv = str(v)
        lines.append(f"{k}: {vv}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_predictions(jsonl_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(jsonl_rows):
        if "imagename" not in row:
            raise ValueError(f"Row {i} missing required key 'imagename'. Keys={list(row.keys())}")
        if "answer" not in row:
            raise ValueError(f"Row {i} missing required key 'answer'. Keys={list(row.keys())}")

        image = row["imagename"]
        answer = row["answer"]

        if not isinstance(image, str):
            raise ValueError(f"Row {i} 'imagename' must be a string, got {type(image)}")
        if not isinstance(answer, str):
            # If it's something else, stringify deterministically
            answer = json.dumps(answer, ensure_ascii=False)

        out.append(
            {
                "image": image,
                "answer": answer.replace("|", "\t"),
                "input_tokens": 0,
                "output_tokens": 0,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_json",
        required=True,
        help="Path to source predictions JSON (e.g., chartqa_preds.json).",
    )
    ap.add_argument(
        "--out_root",
        default="/lab-share/CHIP-Majumder-PNR-e2/Public/thomas/vlm-ensemble-chart-extraction/outputs/predict",
        help="Root directory where a new pred dir will be created.",
    )
    ap.add_argument(
        "--run_name",
        default="",
        help="Optional run folder name. If empty, uses timestamp.",
    )
    ap.add_argument(
        "--input_images_dir",
        default="data/ChartQA Dataset/test/png",
        help="Value to write into config.yaml as input_images_dir.",
    )
    ap.add_argument("--model", default="TinyChart", help="Value to write into config.yaml as model.")
    ap.add_argument("--temperature", type=float, default=0.0, help="Value to write into config.yaml as temperature.")
    args = ap.parse_args()

    input_path = Path(args.input_json)
    out_root = Path(args.out_root)

    if not input_path.exists():
        raise FileNotFoundError(str(input_path))
    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=True)

    def _format_temp(t: float) -> str:
        # 2.0 -> 2p0, 0.75 -> 0p75
        s = f"{t:.3f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")


    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    temp_tag = _format_temp(args.temperature)

    run_name = (
        f"{timestamp}__"
        f"{args.input_images_dir.split('/')[1].replace(' ', '_')}_"
        f"{args.model}_"
        f"temp{temp_tag}_")

    run_name = args.run_name.strip() or run_name
    out_dir = out_root / run_name
    _safe_mkdir(out_dir)

    raw = _read_json(input_path)
    predictions = build_predictions(raw)

    metrics = {
        "num_images": len(predictions),
        "num_processed": len(predictions),
        "num_failed": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    config = {
        "input_images_dir": args.input_images_dir,
        "model": args.model,
        "temperature": float(args.temperature),
    }

    _write_json(out_dir / "metrics.json", metrics)
    _write_yaml_minimal(out_dir / "config.yaml", config)
    _write_json(out_dir / "predictions.json", predictions)

    print(f"Created: {out_dir}")
    print(f"- metrics.json ({len(predictions)} images)")
    print(f"- config.yaml")
    print(f"- predictions.json")


if __name__ == "__main__":
    main()
