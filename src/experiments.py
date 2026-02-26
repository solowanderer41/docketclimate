"""
Experimentation framework for The Docket.

Manages hook variant A/B testing weights using Thompson sampling.
Variant weights are stored in analytics/variant_weights.json and
loaded by content_generator.py when selecting hook variants.

Hook variants (0-4):
    0: Sensory → Data → Stakes
    1: Data-first → Sensory → Stakes
    2: Question → Data → Scene
    3: Moral-stakes — collective failure / accountability
    4: Contrarian-reframe — everyone's looking at X, miss Y

Usage:
    python -m src.main update-weights          # recalculate from metrics
    python -m src.main update-weights --dry    # preview without saving
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

PROJECT_ROOT = Path(__file__).parent.parent
WEIGHTS_PATH = PROJECT_ROOT / "analytics" / "variant_weights.json"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "analytics" / "metrics.json"

VARIANT_NAMES = {
    0: "Sensory→Data→Stakes",
    1: "Data-first",
    2: "Question-led",
    3: "Moral-stakes",
    4: "Contrarian-reframe",
}

# Minimum weight floor: no variant drops below 10% selection probability
MIN_WEIGHT_FLOOR = 0.10
NUM_VARIANTS = 5
MIN_POSTS_FOR_UPDATE = 20  # need at least this many tracked posts to adjust weights


@dataclass
class VariantStats:
    """Aggregated performance stats for a single hook variant."""
    variant_id: int
    count: int
    total_score: float
    avg_score: float
    scores: list[float]


def load_variant_stats(metrics_path: Path = DEFAULT_METRICS_PATH) -> dict[int, VariantStats]:
    """
    Load engagement metrics and compute per-variant stats.

    Returns dict mapping variant_id -> VariantStats.
    """
    from src.analytics import load_metrics

    metrics = load_metrics(metrics_path)
    if not metrics:
        return {}

    variant_data: dict[int, list[float]] = defaultdict(list)
    for m in metrics:
        if m.hook_variant is not None:
            variant_data[m.hook_variant].append(m.engagement_score)

    result = {}
    for vid, scores in variant_data.items():
        n = len(scores)
        result[vid] = VariantStats(
            variant_id=vid,
            count=n,
            total_score=sum(scores),
            avg_score=sum(scores) / n if n else 0,
            scores=scores,
        )

    return result


def thompson_sampling_weights(
    stats: dict[int, VariantStats],
    n_samples: int = 10000,
) -> list[float]:
    """
    Compute variant selection weights using Thompson sampling.

    For each variant, models engagement as a Gaussian with observed mean/std.
    Samples from each variant's posterior and counts how often each wins.
    The win rate becomes the selection weight.

    Variants with no data get a wide uniform prior (high exploration).
    All weights are floored at MIN_WEIGHT_FLOOR to prevent starvation.

    Returns a list of 5 weights (one per variant), summing to ~1.0.
    """
    # Build posterior parameters for each variant
    # Use Gaussian approximation: mean ± std
    posteriors = []
    for vid in range(NUM_VARIANTS):
        if vid in stats and stats[vid].count >= 3:
            s = stats[vid]
            mean = s.avg_score
            # Sample standard deviation (or minimum spread of 1.0)
            if s.count > 1:
                variance = sum((x - mean) ** 2 for x in s.scores) / (s.count - 1)
                std = max(math.sqrt(variance), 1.0)
            else:
                std = max(mean * 0.5, 1.0)
            # Uncertainty decreases with more samples
            std_of_mean = std / math.sqrt(s.count)
            posteriors.append((mean, std_of_mean))
        else:
            # No data or very few samples: wide prior centered at global mean
            all_scores = []
            for s in stats.values():
                all_scores.extend(s.scores)
            global_mean = sum(all_scores) / len(all_scores) if all_scores else 5.0
            posteriors.append((global_mean, global_mean * 0.5 + 1.0))

    # Sample and count wins
    win_counts = [0] * NUM_VARIANTS
    for _ in range(n_samples):
        samples = [random.gauss(mu, sigma) for mu, sigma in posteriors]
        winner = max(range(NUM_VARIANTS), key=lambda i: samples[i])
        win_counts[winner] += 1

    # Convert to weights
    total = sum(win_counts)
    raw_weights = [c / total for c in win_counts]

    # Apply floor: no variant drops below MIN_WEIGHT_FLOOR.
    # Redistribute excess from top performers so the floor holds after norm.
    floor_total = MIN_WEIGHT_FLOOR * NUM_VARIANTS
    if floor_total >= 1.0:
        # Edge case: floor alone uses all budget → equal weights
        return [1.0 / NUM_VARIANTS] * NUM_VARIANTS

    weights = list(raw_weights)
    for _ in range(10):  # iterate to convergence
        deficit = 0.0
        above_floor = []
        for i in range(NUM_VARIANTS):
            if weights[i] < MIN_WEIGHT_FLOOR:
                deficit += MIN_WEIGHT_FLOOR - weights[i]
                weights[i] = MIN_WEIGHT_FLOOR
            else:
                above_floor.append(i)
        if deficit == 0 or not above_floor:
            break
        # Tax the above-floor variants proportionally
        above_sum = sum(weights[i] for i in above_floor)
        for i in above_floor:
            weights[i] -= deficit * (weights[i] / above_sum)

    # Final normalisation for floating-point safety
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    return weights


def save_weights(weights: list[float], stats: dict[int, VariantStats],
                 path: Path = WEIGHTS_PATH):
    """Save variant weights to JSON with metadata."""
    from datetime import datetime

    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "weights": [round(w, 4) for w in weights],
        "updated_at": datetime.now().isoformat(),
        "total_tracked_posts": sum(s.count for s in stats.values()),
        "per_variant": {
            str(vid): {
                "count": stats[vid].count if vid in stats else 0,
                "avg_score": round(stats[vid].avg_score, 2) if vid in stats else 0,
                "weight": round(weights[vid], 4),
                "name": VARIANT_NAMES.get(vid, f"Variant {vid}"),
            }
            for vid in range(NUM_VARIANTS)
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Variant weights saved to {path}[/green]")


def load_weights(path: Path = WEIGHTS_PATH) -> list[float]:
    """Load weights from JSON, or return equal defaults."""
    try:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            w = data.get("weights", [])
            if len(w) == NUM_VARIANTS:
                return w
    except Exception:
        pass
    return [1.0 / NUM_VARIANTS] * NUM_VARIANTS


def update_weights(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    weights_path: Path = WEIGHTS_PATH,
    dry_run: bool = False,
) -> dict:
    """
    Recalculate variant weights from engagement metrics using Thompson sampling.

    Args:
        metrics_path: Path to cumulative metrics JSON.
        weights_path: Path to save updated weights.
        dry_run: If True, display results without saving.

    Returns:
        Dict with old weights, new weights, and per-variant stats.
    """
    stats = load_variant_stats(metrics_path)

    total_tracked = sum(s.count for s in stats.values())
    old_weights = load_weights(weights_path)

    console.print(Panel.fit(
        f"[bold cyan]Hook Variant Weight Update[/bold cyan]\n"
        f"{'DRY RUN — preview only' if dry_run else 'Updating weights'}\n"
        f"{total_tracked} posts with variant data",
        border_style="cyan",
    ))

    if total_tracked < MIN_POSTS_FOR_UPDATE:
        console.print(
            f"[yellow]Need at least {MIN_POSTS_FOR_UPDATE} tracked posts "
            f"to update weights (have {total_tracked}). "
            f"Keeping equal weights.[/yellow]"
        )
        return {
            "status": "insufficient_data",
            "total_tracked": total_tracked,
            "min_required": MIN_POSTS_FOR_UPDATE,
        }

    new_weights = thompson_sampling_weights(stats)

    # Display comparison table
    table = Table(title="Variant Weight Update", show_lines=False)
    table.add_column("Var", width=4, justify="center")
    table.add_column("Strategy", width=24)
    table.add_column("Posts", justify="center", width=7)
    table.add_column("Avg Score", justify="right", width=10)
    table.add_column("Old Weight", justify="right", width=11)
    table.add_column("New Weight", justify="right", width=11)
    table.add_column("Change", justify="right", width=8)

    for vid in range(NUM_VARIANTS):
        name = VARIANT_NAMES.get(vid, f"Variant {vid}")
        count = stats[vid].count if vid in stats else 0
        avg = stats[vid].avg_score if vid in stats else 0
        old_w = old_weights[vid] if vid < len(old_weights) else 0.2
        new_w = new_weights[vid]
        delta = new_w - old_w
        delta_str = f"{'+'if delta >= 0 else ''}{delta:.3f}"
        delta_color = "green" if delta > 0.01 else ("red" if delta < -0.01 else "dim")

        table.add_row(
            str(vid),
            name,
            str(count),
            f"{avg:.1f}" if count else "—",
            f"{old_w:.3f}",
            f"[bold]{new_w:.3f}[/bold]",
            f"[{delta_color}]{delta_str}[/{delta_color}]",
        )

    console.print(table)

    if not dry_run:
        save_weights(new_weights, stats, weights_path)
    else:
        console.print("[dim]Dry run — no changes saved.[/dim]")

    return {
        "status": "updated" if not dry_run else "dry_run",
        "total_tracked": total_tracked,
        "old_weights": old_weights,
        "new_weights": new_weights,
        "per_variant": {
            vid: {
                "count": stats[vid].count if vid in stats else 0,
                "avg_score": stats[vid].avg_score if vid in stats else 0,
            }
            for vid in range(NUM_VARIANTS)
        },
    }
