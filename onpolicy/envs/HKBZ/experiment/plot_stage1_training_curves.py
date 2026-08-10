#!/usr/bin/env python3
"""Plot the recoverable Stage1 training history and held-out test60 endpoints."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


FORMAL_PREFIX = "tail_recovery_stage1_20260730_r1_formal_R3_tail_team_ratio_seed"
SCREEN_PREFIX = "tail_recovery_stage1_20260730_r1_screen_"
DIST_KEYS = {
    "IID": "eval_distribution_iid_makespan",
    "OOD-stress": "eval_distribution_ood_stress_makespan",
    "OOD-scale": "eval_distribution_ood_scale_makespan",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def epoch_from_path(path: Path) -> int | None:
    if path.name == "pre_ppo.json":
        return 0
    match = re.fullmatch(r"epoch_(\d+)\.json", path.name)
    return int(match.group(1)) if match else None


def run_number(path: Path) -> int:
    for part in path.parts:
        match = re.fullmatch(r"run(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def collect_evaluations(experiment_dir: Path) -> dict[int, dict]:
    selected: dict[int, tuple[int, Path]] = {}
    for path in experiment_dir.glob("run*/evaluations/*.json"):
        epoch = epoch_from_path(path)
        if epoch is None:
            continue
        candidate = (run_number(path), path)
        if epoch not in selected or candidate[0] > selected[epoch][0]:
            selected[epoch] = candidate
    return {epoch: load_json(item[1]) for epoch, item in sorted(selected.items())}


def collect_formal(results_root: Path) -> dict[int, dict[int, dict]]:
    histories: dict[int, dict[int, dict]] = {}
    for seed in (1, 2, 3):
        experiment_dir = results_root / f"{FORMAL_PREFIX}{seed}"
        history = collect_evaluations(experiment_dir)
        if history:
            histories[seed] = history
    if not histories:
        raise FileNotFoundError(f"No formal histories found below {results_root}")
    return histories


def collect_screen(results_root: Path) -> dict[str, dict[int, dict]]:
    histories: dict[str, dict[int, dict]] = {}
    for experiment_dir in sorted(results_root.glob(f"{SCREEN_PREFIX}*")):
        match = re.search(r"_screen_(R\d)_", experiment_dir.name)
        if not match:
            continue
        history = collect_evaluations(experiment_dir)
        if history:
            histories[match.group(1)] = history
    return histories


def summary_value(record: dict, key: str) -> float:
    return float(record["summary"][key])


def method_mean(path: Path, method: str) -> float:
    cases = load_json(path)["methods"][method]["cases"]
    makespans = [float(case["makespan"]) for case in cases if case.get("completed", True)]
    if not makespans:
        raise ValueError(f"No completed cases in {path}:{method}")
    return float(np.mean(makespans))


def collect_test60(eval_root: Path) -> list[tuple[str, float, str]]:
    sources = {
        "上一轮 G0": ("fjspv3_test60_trajectory_compare_20260730_previous_g0.json", "DRL-G", "previous"),
        "上一轮 P2": ("fjspv3_test60_trajectory_compare_20260730_current_p2.json", "DRL-G", "previous"),
        "本轮 Seed 1": ("tail_recovery_stage1_20260730_r1_R3_tail_team_ratio_seed1_test60.json", "DRL-G", "current"),
        "本轮 Seed 2": ("tail_recovery_stage1_20260730_r1_R3_tail_team_ratio_seed2_test60.json", "DRL-G", "current"),
        "本轮 Seed 3": ("tail_recovery_stage1_20260730_r1_R3_tail_team_ratio_seed3_test60.json", "DRL-G", "current"),
        "IGA-180": ("fjspv3_test60_trajectory_compare_20260730_iga180.json", "IGA", "iga"),
        "IGA-1800": ("fjspv3_test60_trajectory_compare_20260730_iga1800.json", "IGA", "iga"),
    }
    values: list[tuple[str, float, str]] = []
    current_values: list[float] = []
    for label, (filename, method, group) in sources.items():
        value = method_mean(eval_root / filename, method)
        if group == "current":
            current_values.append(value)
        values.append((label, value, group))
    values.insert(5, ("本轮三种子均值", float(np.mean(current_values)), "current_mean"))
    return values


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Matplotlib 3.6 registers this TTC under its JP family name.  The
            # font still contains the simplified-Chinese and Latin glyphs used
            # by this figure, unlike Droid Sans Fallback (CJK only).
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfc",
            "axes.edgecolor": "#4a4a4a",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
        }
    )


def plot_formal(ax: plt.Axes, formal: dict[int, dict[int, dict]]) -> None:
    colors = {1: "#2878b5", 2: "#e07b39", 3: "#6a9f58"}
    by_epoch: dict[int, list[float]] = defaultdict(list)
    for seed, history in sorted(formal.items()):
        epochs = sorted(history)
        values = [summary_value(history[epoch], "eval_composite_makespan") for epoch in epochs]
        for epoch, value in zip(epochs, values):
            by_epoch[epoch].append(value)
        best_index = int(np.argmin(values))
        ax.plot(epochs, values, "o-", color=colors[seed], alpha=0.70, linewidth=1.6,
                markersize=4.5, label=f"Seed {seed}")
        ax.scatter(epochs[best_index], values[best_index], marker="*", s=105,
                   color=colors[seed], edgecolor="white", linewidth=0.8, zorder=5)

    mean_epochs = sorted(by_epoch)
    means = np.array([np.mean(by_epoch[epoch]) for epoch in mean_epochs])
    stds = np.array([np.std(by_epoch[epoch], ddof=0) for epoch in mean_epochs])
    counts = np.array([len(by_epoch[epoch]) for epoch in mean_epochs])
    complete = counts == 3
    ax.fill_between(np.array(mean_epochs)[complete], (means - stds)[complete],
                    (means + stds)[complete], color="#222222", alpha=0.10, label="三种子 ±1σ")
    ax.plot(mean_epochs[:8], means[:8], "o-", color="#181818", linewidth=2.6,
            markersize=5.5, label="种子均值")
    if len(mean_epochs) > 8:
        ax.plot(mean_epochs[7:], means[7:], "o--", color="#181818", linewidth=2.2,
                markersize=5.5)
        ax.annotate("Epoch 8 仅 Seed 1/2", (mean_epochs[-1], means[-1]), xytext=(-8, -27),
                    textcoords="offset points", ha="right", fontsize=8.5, color="#444444")

    ax.set_title("A  本轮 R3 Formal：Validation Composite Cmax")
    ax.set_xlabel("PPO Epoch（0 = Pre-PPO）")
    ax.set_ylabel("Cmax（越低越好）")
    ax.set_xticks(range(0, 9))
    ax.legend(ncol=2, frameon=True, framealpha=0.92)


def plot_distribution_gain(ax: plt.Axes, formal: dict[int, dict[int, dict]]) -> None:
    colors = {"IID": "#2878b5", "OOD-stress": "#c44e52", "OOD-scale": "#8172b2"}
    line_styles = {"IID": "-", "OOD-stress": "-", "OOD-scale": "--"}
    all_epochs = sorted({epoch for history in formal.values() for epoch in history})
    for label, key in DIST_KEYS.items():
        gain_by_epoch: dict[int, list[float]] = defaultdict(list)
        for history in formal.values():
            if 0 not in history:
                continue
            baseline = summary_value(history[0], key)
            for epoch, record in history.items():
                gain_by_epoch[epoch].append(100.0 * (baseline - summary_value(record, key)) / baseline)
        gains = [np.mean(gain_by_epoch[epoch]) if gain_by_epoch[epoch] else np.nan for epoch in all_epochs]
        ax.plot(all_epochs, gains, marker="o", linestyle=line_styles[label], color=colors[label],
                linewidth=2.1, markersize=4.8, label=label)
    ax.axhline(0.0, color="#555555", linewidth=1.0)
    ax.set_title("B  本轮 R3 Formal：各分布相对 Pre-PPO 改善")
    ax.set_xlabel("PPO Epoch（0 = Pre-PPO）")
    ax.set_ylabel("Cmax 改善（%）")
    ax.set_xticks(range(0, 9))
    ax.legend(frameon=True, framealpha=0.92)


def plot_screen(ax: plt.Axes, screen: dict[str, dict[int, dict]]) -> None:
    colors = {"R0": "#8c8c8c", "R1": "#55a868", "R2": "#4c72b0", "R3": "#c44e52", "R4": "#8172b2"}
    names = {
        "R0": "R0 基线",
        "R1": "R1 Potential",
        "R2": "R2 Team ratio",
        "R3": "R3 Potential + ratio",
        "R4": "R4 + Central critic",
    }
    for config, history in sorted(screen.items()):
        epochs = sorted(history)
        values = [summary_value(history[epoch], "eval_composite_makespan") for epoch in epochs]
        ax.plot(epochs, values, "o-", color=colors.get(config), linewidth=2.0,
                markersize=5.0, label=names.get(config, config))
        if config == "R3":
            ax.annotate(f"最佳 {values[-1]:.1f}", (epochs[-1], values[-1]), xytext=(7, -2),
                        textcoords="offset points", fontsize=8.5, color=colors[config])
    ax.set_title("C  本轮前置 Screen：R0–R4 两个 Epoch")
    ax.set_xlabel("PPO Epoch（0 = Pre-PPO）")
    ax.set_ylabel("Validation Composite Cmax")
    ax.set_xticks([0, 1, 2])
    ax.legend(frameon=True, framealpha=0.92)


def plot_test60(ax: plt.Axes, test60: list[tuple[str, float, str]]) -> None:
    palette = {
        "previous": "#8d99ae",
        "current": "#4c78a8",
        "current_mean": "#173f5f",
        "iga": "#d95f59",
    }
    labels = [item[0] for item in test60]
    values = [item[1] for item in test60]
    colors = [palette[item[2]] for item in test60]
    positions = np.arange(len(test60))
    bars = ax.bar(positions, values, color=colors, width=0.72, edgecolor="white", linewidth=0.8)
    lower = min(values) - 180
    upper = max(values) + 145
    ax.set_ylim(lower, upper)
    ax.set_xticks(positions, labels, rotation=24, ha="right")
    ax.set_ylabel("Test60 mean Cmax（越低越好）")
    ax.set_title("D  上一轮与本轮：Held-out Test60 终点")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:.1f}",
                ha="center", va="bottom", fontsize=8.2, rotation=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.text(0.015, 0.035, "注：上一轮逐 Epoch 日志已清理，图中仅绘制保留下来的真实 test60 终点。",
            transform=ax.transAxes, fontsize=8.3, color="#555555",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result/hkbz_eval_logs/stage1_training_curves_current_vs_previous_20260803.png"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    results_root = root / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo"
    eval_root = root / "result/hkbz_eval_logs"

    formal = collect_formal(results_root)
    screen = collect_screen(results_root)
    test60 = collect_test60(eval_root)

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    plot_formal(axes[0, 0], formal)
    plot_distribution_gain(axes[0, 1], formal)
    plot_screen(axes[1, 0], screen)
    plot_test60(axes[1, 1], test60)
    fig.suptitle("Stage1 OOD 尾部恢复：本轮训练曲线与上一轮结果对照", fontsize=18, fontweight="bold", y=0.972)
    fig.text(
        0.5,
        0.018,
        "数据口径：A–C 为 validation；D 为独立 test60。星号表示各 seed 的最佳 validation checkpoint；Epoch 8 均值仅含完成该轮的 Seed 1/2。",
        ha="center",
        fontsize=9.3,
        color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.10, hspace=0.38, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
