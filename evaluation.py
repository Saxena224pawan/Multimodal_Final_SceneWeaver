"""
Evaluation and Analysis Module for Multi-Agent System

Handles:
- Convergence analysis
- Ablation studies
- Visualization generation
- Comparative metrics
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class ConvergenceStats:
    """Statistics about convergence"""

    total_windows: int = 0
    avg_iterations: float = 0.0
    max_iterations: int = 0
    min_iterations: int = 0
    windows_at_threshold_1: int = 0  # Hit at iteration 1
    windows_at_threshold_2: int = 0  # Hit at iteration 2
    windows_at_threshold_3plus: int = 0  # Hit at iteration 3+
    avg_first_iteration_score: float = 0.0
    avg_final_score: float = 0.0
    score_improvement: float = 0.0  # Final - First iteration
    agent_call_counts: Dict[str, int] = field(default_factory=dict)


def load_metadata(metadata_dir: Path) -> List[Dict[str, Any]]:
    """Load all metadata files from a directory"""
    metadata_files = sorted(metadata_dir.glob("window_*.json"))
    metadatas = []
    for f in metadata_files:
        with open(f) as fp:
            metadatas.append(json.load(fp))
    return metadatas


def compute_convergence_stats(metadatas: List[Dict[str, Any]]) -> ConvergenceStats:
    """Compute convergence statistics from metadata"""
    stats = ConvergenceStats(total_windows=len(metadatas))

    if not metadatas:
        return stats

    iterations_list = []
    first_scores = []
    final_scores = []
    improvements = []

    for m in metadatas:
        iterations = m.get("total_iterations", 0)
        iterations_list.append(iterations)

        scores_history = m.get("scores_history", [])
        if scores_history:
            first_scores.append(scores_history[0])
            final_scores.append(scores_history[-1])
            improvements.append(scores_history[-1] - scores_history[0])

        # Count threshold hits by iteration
        if iterations == 1:
            stats.windows_at_threshold_1 += 1
        elif iterations == 2:
            stats.windows_at_threshold_2 += 1
        else:
            stats.windows_at_threshold_3plus += 1

    stats.avg_iterations = sum(iterations_list) / len(iterations_list) if iterations_list else 0
    stats.max_iterations = max(iterations_list) if iterations_list else 0
    stats.min_iterations = min(iterations_list) if iterations_list else 0
    stats.avg_first_iteration_score = (
        sum(first_scores) / len(first_scores) if first_scores else 0
    )
    stats.avg_final_score = sum(final_scores) / len(final_scores) if final_scores else 0
    stats.score_improvement = sum(improvements) / len(improvements) if improvements else 0

    return stats


def generate_convergence_report(output_dir: Path) -> Dict[str, Any]:
    """Generate comprehensive convergence report"""
    metadata_dir = output_dir / "metadata"

    if not metadata_dir.exists():
        return {"error": "No metadata directory found"}

    metadatas = load_metadata(metadata_dir)
    stats = compute_convergence_stats(metadatas)

    report = {
        "total_windows": stats.total_windows,
        "convergence": {
            "avg_iterations_per_window": round(stats.avg_iterations, 2),
            "max_iterations": stats.max_iterations,
            "min_iterations": stats.min_iterations,
        },
        "threshold_distribution": {
            "hit_at_iteration_1": stats.windows_at_threshold_1,
            "hit_at_iteration_2": stats.windows_at_threshold_2,
            "hit_at_iteration_3_or_more": stats.windows_at_threshold_3plus,
        },
        "scores": {
            "avg_first_iteration": round(stats.avg_first_iteration_score, 3),
            "avg_final": round(stats.avg_final_score, 3),
            "avg_improvement": round(stats.score_improvement, 3),
        },
        "efficiency": {
            "single_pass_success_rate": round(
                stats.windows_at_threshold_1 / stats.total_windows, 2
            )
            if stats.total_windows > 0
            else 0,
            "avg_calls_per_window": round(
                stats.avg_iterations * 3, 0
            ),  # 3 agents per iteration
        },
    }

    return report


def ablate_agent(
    metadatas: List[Dict[str, Any]], agent_name: str
) -> List[Dict[str, Any]]:
    """
    Simulate ablation by removing an agent from the evaluation.

    Returns modified metadata list as if the agent wasn't present.
    """
    agent_key = agent_name.lower()
    abated_metadatas = []

    for m in metadatas:
        abated = dict(m)
        abated_scores_history = []

        for agent_history in m.get("agents_history", []):
            scores = agent_history.get("scores", {})
            # Remove the given agent
            remaining_scores = {k: v for k, v in scores.items() if k != agent_key}

            if remaining_scores:
                avg_score = sum(remaining_scores.values()) / len(remaining_scores)
            else:
                avg_score = 0.5

            abated_scores_history.append(avg_score)

        abated["ablated_agent"] = agent_name
        abated["ablated_scores_history"] = abated_scores_history
        abated_metadatas.append(abated)

    return abated_metadatas


def compute_agent_importance(metadatas: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute agent importance by measuring quality drop when removed.

    Returns: dict mapping agent name to importance score (0-1)
    """
    if not metadatas:
        return {}

    agents = ["continuity", "storybeats", "physics"]
    baseline_score = compute_convergence_stats(metadatas).avg_final_score

    importance = {}
    for agent in agents:
        ablated = ablate_agent(metadatas, agent)
        ablated_stats = compute_convergence_stats(ablated)
        ablated_score = ablated_stats.avg_final_score

        # Importance = how much quality drops without this agent
        drop = baseline_score - ablated_score
        importance[agent] = max(0.0, min(1.0, drop / (baseline_score + 1e-6)))

    return importance


def generate_ablation_report(output_dir: Path) -> Dict[str, Any]:
    """Generate ablation study results"""
    metadata_dir = output_dir / "metadata"

    if not metadata_dir.exists():
        return {"error": "No metadata directory found"}

    metadatas = load_metadata(metadata_dir)
    baseline_stats = compute_convergence_stats(metadatas)
    agent_importance = compute_agent_importance(metadatas)

    report = {
        "baseline_avg_score": round(baseline_stats.avg_final_score, 3),
        "agent_importance": {
            agent: round(importance, 3) for agent, importance in agent_importance.items()
        },
        "ablation_details": {},
    }

    # Add ablation details for each agent
    agents = ["continuity", "storybeats", "physics"]
    for agent in agents:
        ablated = ablate_agent(metadatas, agent)
        ablated_stats = compute_convergence_stats(ablated)

        score_drop = baseline_stats.avg_final_score - ablated_stats.avg_final_score
        report["ablation_details"][agent] = {
            "baseline_score": round(baseline_stats.avg_final_score, 3),
            "ablated_score": round(ablated_stats.avg_final_score, 3),
            "quality_drop": round(score_drop, 3),
            "quality_drop_percent": round((score_drop / baseline_stats.avg_final_score) * 100, 1),
            "impact_level": (
                "HIGH" if score_drop > 0.1 else "MEDIUM" if score_drop > 0.05 else "LOW"
            ),
        }

    return report


def print_convergence_report(report: Dict[str, Any]) -> None:
    """Pretty-print convergence report"""
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS REPORT")
    print("=" * 70 + "\n")

    print(f"Total Windows: {report['total_windows']}\n")

    print("Convergence Statistics:")
    conv = report["convergence"]
    print(f"  Avg iterations/window: {conv['avg_iterations_per_window']}")
    print(f"  Max iterations: {conv['max_iterations']}")
    print(f"  Min iterations: {conv['min_iterations']}\n")

    print("Threshold Distribution:")
    thresh = report["threshold_distribution"]
    print(f"  Hit at iteration 1: {thresh['hit_at_iteration_1']}")
    print(f"  Hit at iteration 2: {thresh['hit_at_iteration_2']}")
    print(f"  Hit at iteration 3+: {thresh['hit_at_iteration_3_or_more']}\n")

    print("Scores:")
    scores = report["scores"]
    print(f"  Avg first iteration: {scores['avg_first_iteration']:.3f}")
    print(f"  Avg final score: {scores['avg_final']:.3f}")
    print(f"  Avg improvement: {scores['avg_improvement']:.3f}\n")

    print("Efficiency:")
    eff = report["efficiency"]
    print(f"  Single-pass success rate: {eff['single_pass_success_rate']:.0%}")
    print(f"  Avg LLM calls/window: {eff['avg_calls_per_window']:.0f}")


def print_ablation_report(report: Dict[str, Any]) -> None:
    """Pretty-print ablation report"""
    print("\n" + "=" * 70)
    print("ABLATION STUDY REPORT")
    print("=" * 70 + "\n")

    print(f"Baseline Avg Score: {report['baseline_avg_score']:.3f}\n")

    print("Agent Importance (0-1, higher = more important):")
    for agent, importance in report["agent_importance"].items():
        star = "⭐" * int(importance * 5)
        print(f"  {agent.capitalize():15} {importance:.3f}  {star}")

    print("\nDetailed Ablation Impact:")
    for agent, details in report["ablation_details"].items():
        print(f"\n  {agent.upper()}:")
        print(f"    Baseline score:     {details['baseline_score']:.3f}")
        print(f"    Ablated score:      {details['ablated_score']:.3f}")
        print(f"    Quality drop:       {details['quality_drop']:.3f} ({details['quality_drop_percent']:.1f}%)")
        print(f"    Impact level:       {details['impact_level']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluation.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])

    # Generate and print reports
    conv_report = generate_convergence_report(output_dir)
    if "error" not in conv_report:
        print_convergence_report(conv_report)

    abl_report = generate_ablation_report(output_dir)
    if "error" not in abl_report:
        print_ablation_report(abl_report)

    # Save reports
    with open(output_dir / "convergence_report.json", "w") as f:
        json.dump(conv_report, f, indent=2)

    with open(output_dir / "ablation_report.json", "w") as f:
        json.dump(abl_report, f, indent=2)

    print("\n✅ Reports saved to:")
    print(f"   {output_dir}/convergence_report.json")
    print(f"   {output_dir}/ablation_report.json\n")
