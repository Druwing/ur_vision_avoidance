#!/usr/bin/env python3
"""Compare YOLO checkpoints on a dataset captured by dataset_capture_node (Table 2).

Every model sees the same images, input size, device and thresholds. Metrics
per model and per ground-truth class: AP@0.5, precision / recall / F1 at the
operating confidence, class-agnostic recall (any predicted class, IoU >= 0.5:
"was the object boxed at all?") and inference latency (mean / p95).

Checkpoints are never downloaded here: a path that does not exist is skipped.
Install the weights manually first, e.g. into ./models/.

Usage:
    python3 scripts/benchmark_yolo.py --dataset /tmp/ur_vision_avoidance_dataset \
        --models yolo26n.pt models/yolo26s.pt models/yolo11n.pt models/yolov8n.pt \
        --alias bench=chair --device cpu --csv /tmp/yolo_benchmark.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, threshold):
    """Greedy matching by confidence. preds: [(conf, box)], gts: [box] -> list of TP flags."""
    used, flags = set(), []
    for _, box in sorted(preds, key=lambda p: -p[0]):
        best, best_iou = None, threshold
        for index, gt in enumerate(gts):
            if index in used:
                continue
            value = iou(box, gt)
            if value >= best_iou:
                best, best_iou = index, value
        flags.append(best is not None)
        if best is not None:
            used.add(best)
    return flags


def average_precision(scored_flags, n_gt) -> float:
    if n_gt == 0:
        return float("nan")
    if not scored_flags:
        return 0.0
    scored_flags = sorted(scored_flags, key=lambda s: -s[0])
    tp = np.cumsum([f for _, f in scored_flags])
    fp = np.cumsum([not f for _, f in scored_flags])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))


def evaluate(model_path, frames, root, args, aliases):
    from ultralytics import YOLO

    model = YOLO(model_path)
    classes = sorted({o["class_name"] for f in frames for o in f["objects"]})
    first = str(root / frames[0]["image"])
    for _ in range(3):
        model.predict(first, conf=args.min_conf, imgsz=args.imgsz, device=args.device, verbose=False)

    latencies, per_image = [], []
    for frame in frames:
        start = time.perf_counter()
        result = model.predict(str(root / frame["image"]), conf=args.min_conf, imgsz=args.imgsz,
                               device=args.device, verbose=False)[0]
        latencies.append((time.perf_counter() - start) * 1e3)
        preds = []
        if result.boxes is not None:
            for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                      result.boxes.cls.cpu().numpy().astype(int)):
                name = str(result.names[int(cls)]).lower()
                preds.append((aliases.get(name, name), float(conf), [float(v) for v in box]))
        per_image.append((frame, preds))

    rows = []
    for cls in classes:
        n_gt, scored, tp_op, fp_op, any_hit = 0, [], 0, 0, 0
        for frame, preds in per_image:
            gts = [o["bbox_xyxy"] for o in frame["objects"] if o["class_name"] == cls]
            n_gt += len(gts)
            cls_preds = [(c, b) for name, c, b in preds if name == cls]
            flags = match(cls_preds, gts, args.iou)
            scored += [(c, f) for (c, _), f in zip(sorted(cls_preds, key=lambda p: -p[0]), flags)]
            op_preds = [(c, b) for c, b in cls_preds if c >= args.conf]
            op_flags = match(op_preds, gts, args.iou)
            tp_op += sum(op_flags)
            fp_op += len(op_flags) - sum(op_flags)
            agnostic = [(c, b) for _, c, b in preds if c >= args.conf]
            any_hit += sum(match(agnostic, gts, args.iou))
        precision = tp_op / (tp_op + fp_op) if tp_op + fp_op else float("nan")
        recall = tp_op / n_gt if n_gt else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if precision and recall and precision + recall else 0.0
        rows.append({
            "model": Path(model_path).name, "class": cls, "n_gt": n_gt,
            "ap50": round(average_precision(scored, n_gt), 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "agnostic_recall": round(any_hit / n_gt, 4) if n_gt else float("nan"),
            "latency_mean_ms": round(float(np.mean(latencies)), 2),
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="output_dir of dataset_capture_node")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conf", type=float, default=0.45, help="operating confidence (vision_node)")
    parser.add_argument("--min-conf", type=float, default=0.01, help="confidence floor for AP")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alias", nargs="*", default=[], help="predicted=ground_truth, e.g. bench=chair")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    root = Path(args.dataset)
    frames = [json.loads(line) for line in (root / "annotations.jsonl").read_text().splitlines() if line.strip()]
    frames = [f for f in frames if f["objects"]]
    if not frames:
        raise SystemExit("no labelled frames in the dataset")
    aliases = dict(item.split("=", 1) for item in args.alias)

    rows = []
    for model_path in args.models:
        if not Path(model_path).exists():
            print(f"SKIP {model_path}: file not found (weights are not downloaded by this script)")
            continue
        print(f"evaluating {model_path} on {len(frames)} frames ...", flush=True)
        rows += evaluate(model_path, frames, root, args, aliases)

    if not rows:
        raise SystemExit("no model evaluated")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nconf={args.conf} iou={args.iou} imgsz={args.imgsz} device={args.device} aliases={aliases or '-'}")
    print("| model | class | n | AP50 | P | R | F1 | class-agnostic R | latency mean / p95 (ms) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['class']} | {r['n_gt']} | {r['ap50']:.3f} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['f1']:.3f} | {r['agnostic_recall']:.3f} | "
              f"{r['latency_mean_ms']:.1f} / {r['latency_p95_ms']:.1f} |")


if __name__ == "__main__":
    main()
