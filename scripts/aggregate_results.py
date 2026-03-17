#!/usr/bin/env python3
"""
Aggregate Results Script - Combine all run results into comprehensive summary

Usage:
    python scripts/aggregate_results.py [output_dir]

Example:
    python scripts/aggregate_results.py outputs/multi_agent_full_run
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def aggregate_convergence_results(output_base: Path) -> Dict[str, Any]:
    """Aggregate convergence reports from all runs"""

    run_dirs = sorted(output_base.glob("run_*"))
    all_reports = {}

    print(f"Found {len(run_dirs)} runs to aggregate...\n")

    for run_dir in run_dirs:
        report_file = run_dir / "convergence_report.json"
        if report_file.exists():
            with open(report_file) as f:
                all_reports[run_dir.name] = json.load(f)
                print(f"  ✓ {run_dir.name}")

    # Compute aggregate statistics
    all_windows = 0
    all_iterations = 0
    all_scores = []
    threshold_hits = 0

    for report in all_reports.values():
        windows = report.get("total_windows", 0)
        all_windows += windows
        all_iterations += report.get("convergence", {}).get("avg_iterations_per_window", 0) * windows
        all_scores.append(report.get("scores", {}).get("avg_final", 0))
        threshold_hits += report.get("threshold_distribution", {}).get("hit_at_iteration_1", 0)

    aggregate = {
        "aggregation_timestamp": Path.cwd().name,
        "total_runs": len(all_reports),
        "total_windows_processed": all_windows,
        "overall_avg_iterations": all_iterations / all_windows if all_windows > 0 else 0,
        "overall_avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "overall_threshold_hit_rate": threshold_hits / all_windows if all_windows > 0 else 0,
        "per_run_results": all_reports,
    }

    return aggregate


def print_summary(aggregate: Dict[str, Any]) -> None:
    """Pretty print aggregate summary"""

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS SUMMARY")
    print("=" * 70 + "\n")

    print("Overall Statistics:")
    print(f"  Total runs: {aggregate['total_runs']}")
    print(f"  Total windows: {aggregate['total_windows_processed']}")
    print(f"  Avg iterations/window: {aggregate['overall_avg_iterations']:.2f}")
    print(f"  Avg quality score: {aggregate['overall_avg_score']:.3f}")
    print(f"  Threshold hit rate: {aggregate['overall_threshold_hit_rate']:.1%}")

    print("\nPer-Run Results:")
    print(f"  {'Run':<20} {'Windows':<10} {'Avg Iter':<10} {'Avg Score':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    for run_name, report in aggregate["per_run_results"].items():
        windows = report.get("total_windows", 0)
        avg_iter = report.get("convergence", {}).get("avg_iterations_per_window", 0)
        avg_score = report.get("scores", {}).get("avg_final", 0)
        print(f"  {run_name:<20} {windows:<10} {avg_iter:<10.2f} {avg_score:<10.3f}")

    print("\n" + "=" * 70)


def main() -> None:
    """Main aggregation function"""

    if len(sys.argv) < 2:
        output_dir = Path("outputs/multi_agent_full_run")
    else:
        output_dir = Path(sys.argv[1])

    if not output_dir.exists():
        print(f"❌ Directory not found: {output_dir}")
        sys.exit(1)

    print(f"\nAggregating results from: {output_dir}\n")

    # Aggregate results
    aggregate = aggregate_convergence_results(output_dir)

    # Print summary
    print_summary(aggregate)

    # Save aggregate file
    aggregate_file = output_dir / "AGGREGATE_RESULTS.json"
    with open(aggregate_file, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\n✅ Aggregate results saved to: {aggregate_file}\n")

    # Print statistics for paper
    print("Statistics for Paper:")
    print(f"  Runs: {aggregate['total_runs']}")
    print(f"  Windows: {aggregate['total_windows_processed']}")
    print(f"  Convergence: {aggregate['overall_avg_iterations']:.2f} iterations")
    print(f"  Quality: {aggregate['overall_avg_score']:.3f} ± ...")
    print(f"  Single-pass success: {aggregate['overall_threshold_hit_rate']:.1%}")
    print()


if __name__ == "__main__":
    main()
