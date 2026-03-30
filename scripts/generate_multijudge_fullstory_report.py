#!/usr/bin/env python3
"""Aggregate full-story prompt results across multiple Video-Bench judge models.

This script reuses the baseline continuity metrics from an existing paper report
directory and pulls prompt metrics from judge-specific benchmark report roots.
It is intended to be run after the additional prompt-judge jobs finish.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DIMENSION_DIRS = {
    "video-text consistency": "video-text_consistency",
    "action": "action",
    "scene": "scene",
    "object_class": "object_class",
    "color": "color",
}

DIMENSION_SHORT = {
    "video-text consistency": "vtc",
    "action": "action",
    "scene": "scene",
    "object_class": "object_class",
    "color": "color",
}

JUDGE_DEFAULTS = (
    ("qwen25vl7b", "Qwen2.5-VL-7B-Instruct", None),
    ("smolvlm", "SmolVLM-Instruct", "benchmark_reports_fullstory_promptjudge_smolvlm"),
    ("llava15_7b", "LLaVA-1.5-7B-HF", "benchmark_reports_fullstory_promptjudge_llava15_7b_sampled3_scored"),
    ("gpt4o_mini", "gpt-4o-mini", "benchmark_reports_fullstory_promptjudge_gpt4o_mini"),
)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_prompt_summary(run_dir: Path, report_root_name: str) -> Optional[Path]:
    root = run_dir / report_root_name / "videobench_window_prompt"
    if not root.is_dir():
        return None
    candidates = sorted(root.glob("videobench_window_prompt_*fullstory/summary.json"))
    return candidates[-1] if candidates else None


def detect_error_summary(dim_dir: Path, payload: Dict[str, object]) -> Optional[str]:
    scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
    if isinstance(scores, dict) and any(str(value).strip().lower() == "error" for value in scores.values()):
        for log_name in ("stdout.log", "stderr.log"):
            log_path = dim_dir / log_name
            if not log_path.is_file():
                continue
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "CUDA out of memory" in text:
                return "CUDA out of memory"
            if "Error code: 500" in text:
                return "server_error"
            if "An error occurred" in text:
                return "judge_error"
        return "judge_error"
    return None


def read_prompt_metrics_from_summary(summary_path: Path) -> Dict[str, object]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results: Dict[str, object] = {
        "status": "ok",
        "error_summary": None,
    }
    for dimension, dim_dir_name in DIMENSION_DIRS.items():
        dim_dir = summary_path.parent / dim_dir_name
        dim_root = dim_dir / "evaluation_results"
        score_files = sorted(dim_root.rglob("*_score_results.json"))
        if not score_files:
            results["status"] = "missing"
            results["error_summary"] = "missing_score_json"
            results["prompt_eval_min"] = float(summary.get("total_elapsed_seconds", 0.0)) / 60.0
            return results
        payload = json.loads(score_files[-1].read_text(encoding="utf-8"))
        avg_scores = payload.get("average_scores", {})
        if not avg_scores:
            results["status"] = "failed"
            results["error_summary"] = detect_error_summary(dim_dir, payload) or "empty_average_scores"
            results["prompt_eval_min"] = float(summary.get("total_elapsed_seconds", 0.0)) / 60.0
            return results
        results[DIMENSION_SHORT[dimension]] = float(next(iter(avg_scores.values())))
    results["prompt_eval_min"] = float(summary.get("total_elapsed_seconds", 0.0)) / 60.0
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-report-dir",
        type=Path,
        required=True,
        help="Paper report directory containing story_metrics.csv and variant_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the multi-judge report files should be written",
    )
    parser.add_argument(
        "--include-gpt",
        action="store_true",
        help="Include the GPT judge slot in the output even if no results are present yet.",
    )
    return parser.parse_args()


def build_judges(include_gpt: bool) -> List[Dict[str, Optional[str]]]:
    judges = []
    for tag, label, report_root_name in JUDGE_DEFAULTS:
        if tag == "gpt4o_mini" and not include_gpt:
            continue
        judges.append(
            {
                "tag": tag,
                "label": label,
                "report_root_name": report_root_name,
            }
        )
    return judges


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_story_rows = load_csv_rows(args.baseline_report_dir / "story_metrics.csv")
    baseline_variant_rows = load_csv_rows(args.baseline_report_dir / "variant_summary.csv")

    judges = build_judges(bool(args.include_gpt))

    baseline_index = {
        (row["variant"], row["story"]): row
        for row in baseline_story_rows
    }
    records: List[Dict[str, object]] = []

    for judge in judges:
        for key, base_row in baseline_index.items():
            variant, story = key
            record = {
                "judge_tag": judge["tag"],
                "judge_label": judge["label"],
                "variant": variant,
                "variant_label": base_row["variant_label"],
                "story": story,
                "story_label": base_row["story_label"],
                "run_dir": base_row["run_dir"],
                "summary_path": None,
                "status": "missing",
                "error_summary": None,
                "vtc": None,
                "action": None,
                "scene": None,
                "object_class": None,
                "color": None,
                "prompt_eval_min": None,
                "subject_consistency": float(base_row["subject_consistency"]),
                "background_consistency": float(base_row["background_consistency"]),
                "motion_smoothness": float(base_row["motion_smoothness"]),
                "temporal_flickering": float(base_row["temporal_flickering"]),
                "continuity_eval_min": float(base_row["continuity_eval_min"]),
            }
            if judge["report_root_name"] is None:
                record.update(
                        {
                            "status": base_row["prompt_status"],
                            "summary_path": base_row["summary_path"],
                            "error_summary": None,
                            "vtc": float(base_row["vtc"]),
                            "action": float(base_row["action"]),
                            "scene": float(base_row["scene"]),
                            "object_class": float(base_row["object_class"]),
                            "color": float(base_row["color"]),
                        "prompt_eval_min": float(base_row["prompt_eval_min"]),
                    }
                )
            else:
                prompt_summary = find_prompt_summary(Path(base_row["run_dir"]), str(judge["report_root_name"]))
                if prompt_summary is not None:
                    metrics = read_prompt_metrics_from_summary(prompt_summary)
                    record.update(
                        {
                            "status": metrics["status"],
                            "summary_path": str(prompt_summary),
                            "error_summary": metrics.get("error_summary"),
                            "prompt_eval_min": metrics["prompt_eval_min"],
                        }
                    )
                    if metrics["status"] == "ok":
                        record.update(
                            {
                                "vtc": metrics["vtc"],
                                "action": metrics["action"],
                                "scene": metrics["scene"],
                                "object_class": metrics["object_class"],
                                "color": metrics["color"],
                            }
                        )
            records.append(record)

    status_groups: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in records:
        status_groups[(row["judge_tag"], row["variant"])].append(row)

    variant_summary: List[Dict[str, object]] = []
    for judge in judges:
        for base_variant in baseline_variant_rows:
            key = (judge["tag"], base_variant["variant"])
            all_rows = status_groups.get(key, [])
            rows = [row for row in all_rows if row["status"] == "ok"]
            variant_summary.append(
                {
                    "judge_tag": judge["tag"],
                    "judge_label": judge["label"],
                    "variant": base_variant["variant"],
                    "variant_label": base_variant["variant_label"],
                    "stories_total": len(rows),
                    "stories_failed": sum(1 for row in all_rows if row["status"] == "failed"),
                    "stories_missing": sum(1 for row in all_rows if row["status"] == "missing"),
                    "mean_vtc": mean(row["vtc"] for row in rows) if rows else None,
                    "mean_action": mean(row["action"] for row in rows) if rows else None,
                    "mean_scene": mean(row["scene"] for row in rows) if rows else None,
                    "mean_object_class": mean(row["object_class"] for row in rows) if rows else None,
                    "mean_color": mean(row["color"] for row in rows) if rows else None,
                    "mean_prompt_eval_min": mean(row["prompt_eval_min"] for row in rows) if rows else None,
                    "mean_subject_consistency": float(base_variant["mean_subject_consistency"]),
                    "mean_background_consistency": float(base_variant["mean_background_consistency"]),
                    "mean_motion_smoothness": float(base_variant["mean_motion_smoothness"]),
                    "mean_temporal_flickering": float(base_variant["mean_temporal_flickering"]),
                    "mean_continuity_eval_min": float(base_variant["mean_continuity_eval_min"]),
                }
            )

    output_json = {
        "variant_summary": variant_summary,
        "story_metrics": records,
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(output_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (args.output_dir / "variant_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(variant_summary[0].keys()))
        writer.writeheader()
        writer.writerows(variant_summary)

    with (args.output_dir / "story_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Multi-Judge Full-Story Benchmark Report",
        "",
        "## Scope",
        "",
        "This report compares Video-Bench prompt metrics across multiple judge models while reusing the same VBench continuity metrics from the baseline full-story evaluation. The continuity benchmark is judge-independent and is therefore copied from the baseline report for every judge row.",
        "",
        "## Judge Coverage",
        "",
        "| Judge | Variants with complete prompt results | Variants with failed prompt results |",
        "| --- | ---: | ---: |",
    ]
    for judge in judges:
        complete = sum(
            1
            for row in variant_summary
            if row["judge_tag"] == judge["tag"] and row["stories_total"] == 4
        )
        failed = sum(
            1
            for row in variant_summary
            if row["judge_tag"] == judge["tag"] and row["stories_failed"] > 0
        )
        lines.append(f"| {judge['label']} | {complete}/5 | {failed}/5 |")

    lines.extend(
        [
            "",
            "## Aggregate Prompt Metrics by Judge",
            "",
        ]
    )
    for judge in judges:
        lines.extend(
            [
                f"### {judge['label']}",
                "",
                "| Variant | Mean VTC (0-5) | Mean Action (0-3) | Mean Scene (0-3) | Mean Object Class (0-3) | Mean Color (0-3) | Mean prompt eval time (min) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        judge_rows = [row for row in variant_summary if row["judge_tag"] == judge["tag"]]
        for row in judge_rows:
            if row["mean_vtc"] is None:
                if row["stories_failed"] > 0:
                    lines.append(f"| {row['variant_label']} | failed | failed | failed | failed | failed | failed |")
                else:
                    lines.append(f"| {row['variant_label']} | pending | pending | pending | pending | pending | pending |")
            else:
                lines.append(
                    "| {variant_label} | {mean_vtc:.4f} | {mean_action:.4f} | {mean_scene:.4f} | {mean_object_class:.4f} | {mean_color:.4f} | {mean_prompt_eval_min:.2f} |".format(
                        **row
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## Continuity Metrics",
            "",
            "The VBench continuity metrics are unchanged by the Video-Bench judge model. Reuse the baseline continuity tables from the existing report when the new prompt-judge results are merged into the paper text.",
            "",
            "## Notes",
            "",
            "- Rows marked `pending` indicate that the corresponding prompt-judge benchmark outputs have not been found yet under the expected report root.",
            "- Rows marked `failed` indicate that prompt outputs were found, but the score JSONs contained no usable `average_scores`.",
            "- In the current local reruns, both `SmolVLM-Instruct` and `LLaVA-1.5-7B-HF` produced prompt-evaluation outputs with judge-side failures rather than usable scores. The saved stdout logs show `CUDA out of memory` errors during evaluation.",
            "- The current script expects judge-specific prompt outputs under each run directory, for example `benchmark_reports_fullstory_promptjudge_smolvlm`.",
        ]
    )
    (args.output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
