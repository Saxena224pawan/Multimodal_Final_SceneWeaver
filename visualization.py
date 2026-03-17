"""
Visualization module for multi-agent system analysis

Generates publication-quality figures:
- Convergence curves
- Agent importance charts
- Score improvement scatter plots
- Iteration distribution histograms
"""

import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_convergence_curve(metadatas: List[Dict[str, Any]], output_file: Path) -> None:
    """Plot convergence curves showing score improvement across iterations"""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not available - skippingvisualization")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, m in enumerate(metadatas[:5]):  # Show first 5 for clarity
        scores = m.get("scores_history", [])
        if scores:
            ax.plot(
                range(1, len(scores) + 1),
                scores,
                marker='o',
                label=f"Window {i} ({m['narrative_beat'][:20]}...)",
                linewidth=2,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Quality Score")
    ax.set_title("Convergence Curves - Quality Improvement Across Iterations")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved convergence plot: {output_file}")


def plot_agent_importance(ablation_report: Dict[str, Any], output_file: Path) -> None:
    """Plot agent importance scores as bar chart"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    agents = list(ablation_report["agent_importance"].keys())
    importance = list(ablation_report["agent_importance"].values())

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(agents, importance, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    # Add value labels on bars
    for bar, val in zip(bars, importance):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    ax.set_ylabel("Importance Score (0-1)")
    ax.set_title("Agent Contribution to Video Quality")
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved agent importance plot: {output_file}")


def plot_iteration_distribution(metadatas: List[Dict[str, Any]], output_file: Path) -> None:
    """Plot histogram of iteration counts"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = [m.get("total_iterations", 0) for m in metadatas]

    ax.hist(iterations, bins=range(1, max(iterations) + 2), edgecolor='black', alpha=0.7, color='#45B7D1')

    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel("Number of Windows")
    ax.set_title("Iteration Distribution - How Many Times Windows Were Refined")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved iteration distribution plot: {output_file}")


def plot_score_improvement(metadatas: List[Dict[str, Any]], output_file: Path) -> None:
    """Scatter plot of score improvements"""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    initial_scores = []
    final_scores = []
    iterations_list = []

    for m in metadatas:
        scores = m.get("scores_history", [])
        if scores:
            initial_scores.append(scores[0])
            final_scores.append(scores[-1])
            iterations_list.append(m.get("total_iterations", 0))

    # Color by iterations
    scatter = ax.scatter(
        initial_scores,
        final_scores,
        c=iterations_list,
        s=100,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=1
    )

    # Add diagonal line (no improvement)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='No improvement')

    ax.set_xlabel("Initial Score (Iteration 1)")
    ax.set_ylabel("Final Score")
    ax.set_title("Quality Improvement: Initial vs Final Scores")
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add colorbar for iterations
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Number of Iterations")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"✅ Saved score improvement plot: {output_file}")


def generate_all_plots(output_dir: Path) -> None:
    """Generate all visualization plots"""
    if not HAS_MATPLOTLIB:
        print("⚠️  Matplotlib not installed - skipping visualizations")
        print("   Install with: pip install matplotlib")
        return

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70 + "\n")

    # Load data
    metadata_dir = output_dir / "metadata"
    if not metadata_dir.exists():
        print("❌ No metadata directory found")
        return

    metadatas = []
    for f in sorted(metadata_dir.glob("window_*.json")):
        with open(f) as fp:
            metadatas.append(json.load(fp))

    # Load reports
    conv_report_file = output_dir / "convergence_report.json"
    abl_report_file = output_dir / "ablation_report.json"

    if not conv_report_file.exists() or not abl_report_file.exists():
        print("⚠️  Report files not found - run evaluation.py first")
        return

    with open(conv_report_file) as f:
        conv_report = json.load(f)

    with open(abl_report_file) as f:
        abl_report = json.load(f)

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate plots
    plot_convergence_curve(metadatas, plots_dir / "convergence.png")
    plot_agent_importance(abl_report, plots_dir / "agent_importance.png")
    plot_iteration_distribution(metadatas, plots_dir / "iteration_distribution.png")
    plot_score_improvement(metadatas, plots_dir / "score_improvement.png")

    print(f"\n✅ All plots saved to: {plots_dir}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <output_dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    generate_all_plots(output_dir)
