import argparse
import glob
import json
import math
import os
import time
from typing import Dict, List, Tuple


def case_id_for(entry: Dict, index: int) -> str:
    return str(entry.get("case_id") or entry.get("id") or index)


def dataset_name_for(json_file: str) -> str:
    return os.path.splitext(os.path.basename(json_file))[0]


def resolve_loss_dir(args) -> str:
    return args.loss_jsonl_path or os.path.join(args.sum_path, "loss_jsonl")


def select_entries(data: List[Dict], sample_ratio: float) -> List[Tuple[int, Dict]]:
    count = max(1, int(len(data) * sample_ratio)) if data else 0
    return list(enumerate(data[:count]))


def load_summary(sum_path: str) -> Dict[str, Dict]:
    summary = {}
    for sum_file in glob.glob(os.path.join(sum_path, "*_mean.json")):
        with open(sum_file, "r", encoding="utf-8") as f:
            item = json.load(f)
        summary[item["dataset"]] = item
    return summary


def load_loss_jsonl(path: str) -> Dict[str, float]:
    losses = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"loss jsonl not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if len(item) != 1:
                raise ValueError(f"{path}:{line_no} must be a one-item dict like {{case_id: loss}}")
            case_id, loss = next(iter(item.items()))
            losses[str(case_id)] = float(loss)
    return losses


def main(args):
    summary = load_summary(args.sum_path)
    loss_dir = resolve_loss_dir(args)
    reverse = []

    for json_file in glob.glob(os.path.join(args.file_path, "*.json")):
        start_time = time.time()
        dataset_name = dataset_name_for(json_file)
        if dataset_name not in summary:
            raise KeyError(f"missing mean summary for dataset: {dataset_name}")

        dataset_mean_loss = float(summary[dataset_name]["mean"])
        if math.isnan(dataset_mean_loss):
            print(f"skip {dataset_name}: mean loss is NaN")
            continue

        loss_path = summary[dataset_name].get("loss_jsonl") or os.path.join(loss_dir, f"{dataset_name}_loss.jsonl")
        losses = load_loss_jsonl(loss_path)

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        selected = select_entries(data, args.sample_ratio)
        matched = 0
        missing = 0
        for index, entry in selected:
            case_id = case_id_for(entry, index)
            loss = losses.get(case_id)
            if loss is None:
                missing += 1
                continue
            if abs(loss - dataset_mean_loss) <= args.threshold:
                reverse.append(entry)
                matched += 1

        print(
            f"finishing the dataset: {dataset_name}, matched={matched}, "
            f"missing_loss={missing}, time={time.time() - start_time:.2f}s"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(reverse, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=str, required=True)
    parser.add_argument("--sum-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--loss-jsonl-path", type=str, default=None)
    parser.add_argument("--sample-ratio", type=float, default=1)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--modality", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(args)
