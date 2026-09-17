#!/usr/bin/env python3
"""Compare Stage-1 V2 N1/N2 three-seed blind-test results with IGA baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats


METHOD = "DRL-G"
IGA_METHOD = "IGA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n1", nargs=3, required=True, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--n2", nargs=3, required=True, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--n1-validation", nargs=3, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--n2-validation", nargs=3, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--iga180", required=True)
    parser.add_argument("--iga1800", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    return parser.parse_args()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_cases(payload: dict, method: str) -> dict[str, dict]:
    block = payload.get("methods", {}).get(method, {})
    cases = block.get("cases", [])
    if payload.get("status") != "completed" or block.get("status") != "completed":
        raise RuntimeError(f"{method} result is not completed")
    if len(cases) != 60 or not all(case.get("completed") for case in cases):
        raise RuntimeError(f"{method} must contain 60 completed cases; got {len(cases)}")
    if method == IGA_METHOD and not all(case.get("completion_verified") for case in cases):
        raise RuntimeError("IGA must contain 60 authority-replay-verified cases")
    result = {case["case"]: case for case in cases}
    if len(result) != 60:
        raise RuntimeError(f"{method} contains duplicate case names")
    return result


def completed_validation_cases(payload: dict) -> dict[str, dict]:
    cases = payload.get("cases", [])
    if len(cases) != 60 or not all(case.get("completed") for case in cases):
        raise RuntimeError(f"validation result must contain 60 completed cases; got {len(cases)}")
    result = {case.get("case_key", case["case_dir"]): case for case in cases}
    if len(result) != 60:
        raise RuntimeError("validation result contains duplicate case keys")
    return result


def vector_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def paired_stats(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    nonzero = delta[delta != 0.0]
    sample_std = float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
    return {
        "n": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "relative_delta_percent": float(100.0 * delta.mean() / np.asarray(right).mean()),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wins": int(np.sum(delta < 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "losses": int(np.sum(delta > 0.0)),
        "cohens_d": float(delta.mean() / sample_std) if sample_std > 0.0 else 0.0,
        "t_p": float(stats.ttest_1samp(delta, 0.0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0,
        "gap_gt_800": int(np.sum(delta > 800.0)),
        "lead_gt_800": int(np.sum(delta < -800.0)),
        "max_gap": float(delta.max()),
        "max_lead": float(-delta.min()),
    }


def hierarchical_ci(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> list[float]:
    """Resample matched training seeds and cases to cover both uncertainty sources."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 1:
        left = left[None, :]
    if right.ndim == 1:
        right = right[None, :]
    chunks = []
    chunk_size = 10_000
    for start in range(0, samples, chunk_size):
        count = min(chunk_size, samples - start)
        case_idx = rng.integers(0, left.shape[1], size=(count, left.shape[1]))
        seed_idx = rng.integers(0, max(left.shape[0], right.shape[0]), size=(count, 3))
        values = np.empty(count, dtype=float)
        for index in range(count):
            cases = case_idx[index]
            left_seeds = seed_idx[index] % left.shape[0]
            right_seeds = seed_idx[index] % right.shape[0]
            left_mean = left[left_seeds][:, cases].mean()
            right_mean = right[right_seeds][:, cases].mean()
            values[index] = left_mean - right_mean
        chunks.append(values)
    boot = np.concatenate(chunks)
    return [float(x) for x in np.quantile(boot, [0.025, 0.975])]


def group_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output = {}
    for name, items in sorted(groups.items()):
        output[name] = {
            "n": len(items),
            "n1_mean": float(np.mean([item["n1_mean"] for item in items])),
            "n2_mean": float(np.mean([item["n2_mean"] for item in items])),
            "iga180_mean": float(np.mean([item["iga180"] for item in items])),
            "iga1800_mean": float(np.mean([item["iga1800"] for item in items])),
            "n1_minus_n2": float(np.mean([item["delta_n1_n2"] for item in items])),
            "n1_minus_iga1800": float(np.mean([item["delta_n1_iga1800"] for item in items])),
            "n2_minus_iga1800": float(np.mean([item["delta_n2_iga1800"] for item in items])),
        }
    return output


def group_paired_statistics(
    rows: list[dict], key: str, rng: np.random.Generator, samples: int
) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output = {}
    for name, items in sorted(groups.items()):
        n1 = np.asarray([item["n1_mean"] for item in items], dtype=float)
        n2 = np.asarray([item["n2_mean"] for item in items], dtype=float)
        iga1800 = np.asarray([item["iga1800"] for item in items], dtype=float)
        output[name] = {
            "n1_vs_n2": paired_stats(n1, n2, rng, samples),
            "n1_vs_iga1800": paired_stats(n1, iga1800, rng, samples),
            "n2_vs_iga1800": paired_stats(n2, iga1800, rng, samples),
        }
    return output


def seed_stability(matrix: np.ndarray, baseline: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    wins = matrix < baseline[None, :]
    ties = matrix == baseline[None, :]
    ranges = np.ptp(matrix, axis=0)
    return {
        "all_three_beat_or_tie_iga1800": int(np.sum(np.all(wins | ties, axis=0))),
        "all_three_lose_iga1800": int(np.sum(np.all(~wins & ~ties, axis=0))),
        "mixed_vs_iga1800": int(np.sum(np.any(wins, axis=0) & np.any(~wins & ~ties, axis=0))),
        "mean_case_range": float(ranges.mean()),
        "median_case_range": float(np.median(ranges)),
        "p95_case_range": float(np.quantile(ranges, 0.95)),
        "max_case_range": float(ranges.max()),
    }


def fmt(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):{'+' if signed else ''}.{digits}f}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Stage1 V2 N1/N2 三种子 blind test 与 IGA 同案例报告",
        "",
        "Cmax 越低越好；N1/N2 均使用各 seed 的预选最佳 validation checkpoint。",
        "",
        "## 总体结果",
        "",
        "| 方法 | 60例均值 | seed间均值±样本SD | 案例SD | 中位数 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "n1": "N1 三种子案例均值",
        "n2": "N2 三种子案例均值",
        "iga180": "IGA-180",
        "iga1800": "IGA-1800",
    }
    for key in ("n1", "n2", "iga180", "iga1800"):
        row = report["method_stats"][key]
        seed_text = "—"
        if key in ("n1", "n2"):
            seed = report["three_seed_summary"][key]
            seed_text = f"{seed['mean']:.2f}±{seed['sample_std']:.2f}"
        lines.append(
            f"| {labels[key]} | {row['mean']:.2f} | {seed_text} | {row['std']:.2f} | "
            f"{row['median']:.1f} | {row['min']:.0f} | {row['max']:.0f} |"
        )
    lines += ["", "各 seed 的 test60 均值：", ""]
    for method in ("n1", "n2"):
        means = report["three_seed_summary"][method]["seed_means"]
        lines.append(f"- {method.upper()}: " + ", ".join(f"{x:.2f}" for x in means))

    lines += [
        "",
        "## 同案例配对统计",
        "",
        "| 比较（左−右） | 均值差 | 相对差 | 案例bootstrap 95% CI | 层级bootstrap 95% CI | 胜/平/负 | Wilcoxon p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    comparisons = (
        ("n1_vs_n2", "N1−N2"),
        ("n1_vs_iga180", "N1−IGA-180"),
        ("n1_vs_iga1800", "N1−IGA-1800"),
        ("n2_vs_iga180", "N2−IGA-180"),
        ("n2_vs_iga1800", "N2−IGA-1800"),
        ("iga1800_vs_iga180", "IGA-1800−IGA-180"),
    )
    for key, label in comparisons:
        row = report["paired_statistics"][key]
        hierarchical = row.get("hierarchical_ci95")
        hierarchical_text = (
            f"[{fmt(hierarchical[0], 1, True)}, {fmt(hierarchical[1], 1, True)}]"
            if hierarchical else "—"
        )
        lines.append(
            f"| {label} | {fmt(row['mean_delta'], 2, True)} | {fmt(row['relative_delta_percent'], 3, True)}% | "
            f"[{fmt(row['ci95'][0], 1, True)}, {fmt(row['ci95'][1], 1, True)}] | {hierarchical_text} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['wilcoxon_p']:.4g} |"
        )

    goals = report["same_dataset_goals"]
    lines += [
        "",
        "## 同数据集目标判定",
        "",
        f"- 新 test60 的 IGA-1800 均值：{goals['iga1800_mean']:.2f}；103% 接近门槛：{goals['near_iga_threshold']:.2f}。",
        f"- N1：接近 IGA {'通过' if goals['n1_near_iga'] else '未通过'}；均值达到 IGA {'通过' if goals['n1_reach_iga'] else '未通过'}；含 OOD-stress 约束的严格目标 {'通过' if goals['n1_reach_iga_strict'] else '未通过'}。",
        f"- N2：接近 IGA {'通过' if goals['n2_near_iga'] else '未通过'}；均值达到 IGA {'通过' if goals['n2_reach_iga'] else '未通过'}；含 OOD-stress 约束的严格目标 {'通过' if goals['n2_reach_iga_strict'] else '未通过'}。",
        "",
        "## 同 seed 的 N1/N2 比较",
        "",
        "| Seed | N1 | N2 | N1−N2 | 案例bootstrap 95% CI | 胜/平/负 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, row in report["seedwise_method_comparison"].items():
        lines.append(
            f"| {seed} | {row['n1_mean']:.2f} | {row['n2_mean']:.2f} | {fmt(row['paired']['mean_delta'], 2, True)} | "
            f"[{fmt(row['paired']['ci95'][0], 1, True)}, {fmt(row['paired']['ci95'][1], 1, True)}] | "
            f"{row['paired']['wins']}/{row['paired']['ties']}/{row['paired']['losses']} |"
        )

    transfer = report.get("validation_blind_transfer")
    if transfer:
        lines += [
            "",
            "## Validation → blind 泛化迁移",
            "",
            f"Validation 最佳 checkpoint 的 N1−N2 composite 差为 {transfer['validation_composite_delta']:+.1f}；blind test 差为 {transfer['blind_overall_delta']:+.1f}。",
            "",
            "| 分组 | validation N1−N2 | blind N1−N2 |",
            "|---|---:|---:|",
        ]
        for name, row in transfer["distribution"].items():
            lines.append(f"| {name} | {row['validation_delta']:+.1f} | {row['blind_delta']:+.1f} |")
        lines += ["", "重点 profile：", "", "| Profile | validation N1−N2 | blind N1−N2 |", "|---|---:|---:|"]
        for name, row in transfer["profile"].items():
            lines.append(f"| {name} | {row['validation_delta']:+.1f} | {row['blind_delta']:+.1f} |")

    for title, key in (("分布", "distribution_summary"), ("Profile", "profile_summary")):
        lines += [
            "",
            f"## {title}结果",
            "",
            f"| {title} | n | N1 | N2 | IGA-180 | IGA-1800 | N1−N2 | N1−IGA1800 | N2−IGA1800 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in report[key].items():
            lines.append(
                f"| {name} | {row['n']} | {fmt(row['n1_mean'])} | {fmt(row['n2_mean'])} | "
                f"{fmt(row['iga180_mean'])} | {fmt(row['iga1800_mean'])} | "
                f"{fmt(row['n1_minus_n2'], 1, True)} | {fmt(row['n1_minus_iga1800'], 1, True)} | "
                f"{fmt(row['n2_minus_iga1800'], 1, True)} |"
            )

    lines += ["", "## 尾部与 seed 稳定性", ""]
    for method in ("n1", "n2"):
        tail = report["tail_summary"][method]
        stable = report["seed_stability"][method]
        lines += [
            f"### {method.upper()}",
            "",
            f"- 相对 IGA-1800 gap>800：{tail['severe_count']} 个；正差距中占比 {tail['severe_share_of_positive_gap'] * 100:.1f}%。",
            f"- policy worst-6 均值：{tail['policy_worst6_mean']:.1f}；对应 IGA-1800 均值：{tail['iga1800_on_policy_worst6_mean']:.1f}。",
            f"- 三 seed 全部优于/不差于 IGA-1800：{stable['all_three_beat_or_tie_iga1800']}；全部落后：{stable['all_three_lose_iga1800']}；胜负混合：{stable['mixed_vs_iga1800']}。",
            f"- 单案例 seed 极差：均值 {stable['mean_case_range']:.1f}，P95 {stable['p95_case_range']:.1f}，最大 {stable['max_case_range']:.1f}。",
            "",
        ]

    concentration = report["n1_n2_concentration"]
    lines += [
        "### N1 相对 N2 的收益集中度",
        "",
        f"- N1 的全部逐案例领先量为 {concentration['n1_total_lead']:.1f}，其中最大两个领先案例贡献 {concentration['top2_lead_share'] * 100:.1f}%。",
        f"- 这两个案例贡献 N1 在 OOD-stress 上净优势的 {concentration['top2_share_of_ood_stress_net_advantage'] * 100:.1f}%。",
        f"- 去掉这两个案例后，N1−N2 的其余 58 例均值为 {concentration['mean_delta_without_top2']:+.1f}。",
        f"- 最大两个领先案例：{', '.join(concentration['top2_lead_cases'])}。",
        "",
    ]

    lines += [
        "## 相对 IGA-1800 最差的 10 个案例",
        "",
        "| Method | Case | Profile | Dist | 模型均值 | IGA-1800 | Gap |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for method in ("n1", "n2"):
        for row in report["worst_vs_iga1800"][method]:
            lines.append(
                f"| {method.upper()} | {row['case']} | {row['profile']} | {row['distribution']} | "
                f"{fmt(row[f'{method}_mean'])} | {fmt(row['iga1800'], 0)} | "
                f"{fmt(row[f'delta_{method}_iga1800'], 1, True)} |"
            )

    lines += [
        "",
        "## 60 个案例逐例结果",
        "",
        "| Case | Profile | Dist | N1-S1 | N1-S2 | N1-S3 | N1均值 | N2-S1 | N2-S2 | N2-S3 | N2均值 | IGA180 | IGA1800 | N1−IGA1800 | N2−IGA1800 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case']} | {row['profile']} | {row['distribution']} | "
            f"{fmt(row['n1_seed1'], 0)} | {fmt(row['n1_seed2'], 0)} | {fmt(row['n1_seed3'], 0)} | {fmt(row['n1_mean'])} | "
            f"{fmt(row['n2_seed1'], 0)} | {fmt(row['n2_seed2'], 0)} | {fmt(row['n2_seed3'], 0)} | {fmt(row['n2_mean'])} | "
            f"{fmt(row['iga180'], 0)} | {fmt(row['iga1800'], 0)} | "
            f"{fmt(row['delta_n1_iga1800'], 1, True)} | {fmt(row['delta_n2_iga1800'], 1, True)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    n1_payloads = [read_json(path) for path in args.n1]
    n2_payloads = [read_json(path) for path in args.n2]
    if bool(args.n1_validation) != bool(args.n2_validation):
        raise ValueError("--n1-validation and --n2-validation must be provided together")
    iga180_payload = read_json(args.iga180)
    iga1800_payload = read_json(args.iga1800)
    n1_maps = [completed_cases(payload, METHOD) for payload in n1_payloads]
    n2_maps = [completed_cases(payload, METHOD) for payload in n2_payloads]
    iga180 = completed_cases(iga180_payload, IGA_METHOD)
    iga1800 = completed_cases(iga1800_payload, IGA_METHOD)

    case_names = sorted(iga1800)
    expected = set(case_names)
    for name, mapping in [
        ("IGA-180", iga180),
        *[(f"N1 seed{i + 1}", value) for i, value in enumerate(n1_maps)],
        *[(f"N2 seed{i + 1}", value) for i, value in enumerate(n2_maps)],
    ]:
        if set(mapping) != expected:
            raise RuntimeError(f"case mismatch for {name}")

    rows = []
    for case in case_names:
        records = [iga180[case], iga1800[case]]
        records.extend(mapping[case] for mapping in n1_maps + n2_maps)
        hashes = {record.get("case_sha256") for record in records}
        if len(hashes) != 1:
            raise RuntimeError(f"case integrity mismatch for {case}: {hashes}")
        n1 = np.asarray([mapping[case]["makespan"] for mapping in n1_maps], dtype=float)
        n2 = np.asarray([mapping[case]["makespan"] for mapping in n2_maps], dtype=float)
        i180 = float(iga180[case]["makespan"])
        i1800 = float(iga1800[case]["makespan"])
        rows.append({
            "case": case,
            "case_id": iga1800[case].get("case_id"),
            "case_sha256": iga1800[case].get("case_sha256"),
            "profile": iga1800[case].get("profile", "unknown"),
            "distribution": iga1800[case].get("distribution", "unknown"),
            "n1_seed1": float(n1[0]), "n1_seed2": float(n1[1]), "n1_seed3": float(n1[2]),
            "n1_mean": float(n1.mean()), "n1_sample_std": float(n1.std(ddof=1)), "n1_range": float(np.ptp(n1)),
            "n2_seed1": float(n2[0]), "n2_seed2": float(n2[1]), "n2_seed3": float(n2[2]),
            "n2_mean": float(n2.mean()), "n2_sample_std": float(n2.std(ddof=1)), "n2_range": float(np.ptp(n2)),
            "iga180": i180, "iga1800": i1800,
            "delta_n1_n2": float(n1.mean() - n2.mean()),
            "delta_n1_iga180": float(n1.mean() - i180),
            "delta_n1_iga1800": float(n1.mean() - i1800),
            "delta_n2_iga180": float(n2.mean() - i180),
            "delta_n2_iga1800": float(n2.mean() - i1800),
        })

    n1_matrix = np.asarray([[row[f"n1_seed{i}"] for row in rows] for i in range(1, 4)])
    n2_matrix = np.asarray([[row[f"n2_seed{i}"] for row in rows] for i in range(1, 4)])
    n1_mean = n1_matrix.mean(axis=0)
    n2_mean = n2_matrix.mean(axis=0)
    i180 = np.asarray([row["iga180"] for row in rows])
    i1800 = np.asarray([row["iga1800"] for row in rows])
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = {
        "n1_vs_n2": (n1_mean, n2_mean, n1_matrix, n2_matrix),
        "n1_vs_iga180": (n1_mean, i180, n1_matrix, i180),
        "n1_vs_iga1800": (n1_mean, i1800, n1_matrix, i1800),
        "n2_vs_iga180": (n2_mean, i180, n2_matrix, i180),
        "n2_vs_iga1800": (n2_mean, i1800, n2_matrix, i1800),
        "iga1800_vs_iga180": (i1800, i180, i1800, i180),
    }
    paired = {}
    for name, (left, right, left_matrix, right_matrix) in comparisons.items():
        paired[name] = paired_stats(left, right, rng, args.bootstrap_samples)
        if name != "iga1800_vs_iga180":
            paired[name]["hierarchical_ci95"] = hierarchical_ci(
                left_matrix, right_matrix, rng, args.bootstrap_samples
            )

    seed_means = {
        "n1": n1_matrix.mean(axis=1),
        "n2": n2_matrix.mean(axis=1),
    }
    seedwise = {}
    for seed_index in range(3):
        seedwise[str(seed_index + 1)] = {
            "n1_mean": float(n1_matrix[seed_index].mean()),
            "n2_mean": float(n2_matrix[seed_index].mean()),
            "paired": paired_stats(
                n1_matrix[seed_index], n2_matrix[seed_index], rng,
                args.bootstrap_samples,
            ),
        }

    def tail_summary(method: str, values: np.ndarray) -> dict:
        deltas = values - i1800
        severe = deltas > 800.0
        positive_sum = float(np.maximum(deltas, 0.0).sum())
        worst_idx = np.argsort(values)[-6:]
        return {
            "severe_count": int(severe.sum()),
            "severe_cases": [rows[index]["case"] for index in np.flatnonzero(severe)],
            "severe_gap_sum": float(deltas[severe].sum()),
            "positive_gap_sum": positive_sum,
            "severe_share_of_positive_gap": float(deltas[severe].sum() / positive_sum) if positive_sum else 0.0,
            "policy_worst6_cases": [rows[index]["case"] for index in worst_idx],
            "policy_worst6_mean": float(values[worst_idx].mean()),
            "iga1800_on_policy_worst6_mean": float(i1800[worst_idx].mean()),
            "net_gap_sum": float(deltas.sum()),
        }

    iga_mean = float(i1800.mean())
    stress_mask = np.asarray([row["distribution"] == "ood_stress" for row in rows])
    iga_stress_mean = float(i1800[stress_mask].mean())
    n1_stress_mean = float(n1_mean[stress_mask].mean())
    n2_stress_mean = float(n2_mean[stress_mask].mean())
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_unix_time": time.time(),
        "sources": {
            "n1": [str(Path(path).resolve()) for path in args.n1],
            "n2": [str(Path(path).resolve()) for path in args.n2],
            "iga180": str(Path(args.iga180).resolve()),
            "iga1800": str(Path(args.iga1800).resolve()),
            "n1_validation": [str(Path(path).resolve()) for path in args.n1_validation] if args.n1_validation else None,
            "n2_validation": [str(Path(path).resolve()) for path in args.n2_validation] if args.n2_validation else None,
        },
        "integrity": {"case_count": 60, "all_case_names_and_hashes_match": True},
        "method_stats": {
            "n1": vector_summary(n1_mean), "n2": vector_summary(n2_mean),
            "iga180": vector_summary(i180), "iga1800": vector_summary(i1800),
        },
        "three_seed_summary": {
            method: {
                "seed_means": [float(x) for x in means],
                "mean": float(means.mean()),
                "sample_std": float(means.std(ddof=1)),
            }
            for method, means in seed_means.items()
        },
        "same_dataset_goals": {
            "iga1800_mean": iga_mean,
            "iga1800_ood_stress_mean": iga_stress_mean,
            "near_iga_threshold": 1.03 * iga_mean,
            "n1_near_iga": bool(n1_mean.mean() <= 1.03 * iga_mean),
            "n1_reach_iga": bool(n1_mean.mean() <= iga_mean),
            "n1_ood_stress_mean": n1_stress_mean,
            "n1_ood_stress_not_worse": bool(n1_stress_mean <= iga_stress_mean),
            "n1_reach_iga_strict": bool(n1_mean.mean() <= iga_mean and n1_stress_mean <= iga_stress_mean),
            "n2_near_iga": bool(n2_mean.mean() <= 1.03 * iga_mean),
            "n2_reach_iga": bool(n2_mean.mean() <= iga_mean),
            "n2_ood_stress_mean": n2_stress_mean,
            "n2_ood_stress_not_worse": bool(n2_stress_mean <= iga_stress_mean),
            "n2_reach_iga_strict": bool(n2_mean.mean() <= iga_mean and n2_stress_mean <= iga_stress_mean),
        },
        "paired_statistics": paired,
        "seedwise_method_comparison": seedwise,
        "distribution_summary": group_summary(rows, "distribution"),
        "profile_summary": group_summary(rows, "profile"),
        "group_paired_statistics": {
            "distribution": group_paired_statistics(
                rows, "distribution", rng, args.bootstrap_samples
            ),
            "profile": group_paired_statistics(
                rows, "profile", rng, args.bootstrap_samples
            ),
        },
        "seed_stability": {
            "n1": seed_stability(n1_matrix, i1800),
            "n2": seed_stability(n2_matrix, i1800),
        },
        "tail_summary": {
            "n1": tail_summary("n1", n1_mean),
            "n2": tail_summary("n2", n2_mean),
        },
        "failure_migration_n1_to_n2": {},
        "worst_vs_iga1800": {
            "n1": sorted(rows, key=lambda row: row["delta_n1_iga1800"], reverse=True)[:10],
            "n2": sorted(rows, key=lambda row: row["delta_n2_iga1800"], reverse=True)[:10],
        },
        "largest_n1_n2_disagreements": sorted(rows, key=lambda row: abs(row["delta_n1_n2"]), reverse=True)[:15],
        "cases": rows,
    }

    if args.n1_validation:
        n1_validation_payloads = [read_json(path) for path in args.n1_validation]
        n2_validation_payloads = [read_json(path) for path in args.n2_validation]
        n1_validation_maps = [completed_validation_cases(payload) for payload in n1_validation_payloads]
        n2_validation_maps = [completed_validation_cases(payload) for payload in n2_validation_payloads]
        validation_keys = sorted(n1_validation_maps[0])
        expected_validation = set(validation_keys)
        for mapping in n1_validation_maps + n2_validation_maps:
            if set(mapping) != expected_validation:
                raise RuntimeError("validation case mismatch between N1 and N2")
        validation_rows = []
        for key in validation_keys:
            records = [mapping[key] for mapping in n1_validation_maps + n2_validation_maps]
            hashes = {record.get("case_sha256") for record in records}
            if len(hashes) != 1:
                raise RuntimeError(f"validation integrity mismatch for {key}: {hashes}")
            n1_value = float(np.mean([mapping[key]["makespan"] for mapping in n1_validation_maps]))
            n2_value = float(np.mean([mapping[key]["makespan"] for mapping in n2_validation_maps]))
            validation_rows.append({
                "key": key,
                "profile": records[0].get("profile", "unknown"),
                "distribution": records[0].get("distribution", "unknown"),
                "delta": n1_value - n2_value,
            })

        def transfer_groups(validation_key: str, blind_key: str) -> dict:
            names = sorted({str(row[validation_key]) for row in validation_rows})
            output = {}
            for name in names:
                validation_delta = float(np.mean([
                    row["delta"] for row in validation_rows if str(row[validation_key]) == name
                ]))
                blind_delta = float(np.mean([
                    row["delta_n1_n2"] for row in rows if str(row[blind_key]) == name
                ]))
                output[name] = {
                    "validation_delta": validation_delta,
                    "blind_delta": blind_delta,
                }
            return output

        validation_n1_composite = float(np.mean([
            payload["summary"]["eval_composite_makespan"] for payload in n1_validation_payloads
        ]))
        validation_n2_composite = float(np.mean([
            payload["summary"]["eval_composite_makespan"] for payload in n2_validation_payloads
        ]))
        report["validation_blind_transfer"] = {
            "validation_n1_composite": validation_n1_composite,
            "validation_n2_composite": validation_n2_composite,
            "validation_composite_delta": validation_n1_composite - validation_n2_composite,
            "blind_overall_delta": float(n1_mean.mean() - n2_mean.mean()),
            "distribution": transfer_groups("distribution", "distribution"),
            "profile": transfer_groups("profile", "profile"),
        }
    n1_leads = sorted(
        (row for row in rows if row["delta_n1_n2"] < 0.0),
        key=lambda row: row["delta_n1_n2"],
    )
    n1_total_lead = float(-sum(row["delta_n1_n2"] for row in n1_leads))
    top2_cases = {row["case"] for row in n1_leads[:2]}
    top2_lead = float(-sum(row["delta_n1_n2"] for row in n1_leads[:2]))
    remaining = [row["delta_n1_n2"] for row in rows if row["case"] not in top2_cases]
    report["n1_n2_concentration"] = {
        "n1_total_lead": n1_total_lead,
        "n1_total_cost": float(sum(max(0.0, row["delta_n1_n2"]) for row in rows)),
        "top2_lead_cases": [row["case"] for row in n1_leads[:2]],
        "top2_lead": top2_lead,
        "top2_lead_share": top2_lead / n1_total_lead if n1_total_lead else 0.0,
        "top2_share_of_ood_stress_net_advantage": top2_lead / -sum(
            row["delta_n1_n2"] for row in rows if row["distribution"] == "ood_stress"
        ),
        "mean_delta_without_top2": float(np.mean(remaining)),
    }
    n1_severe = set(report["tail_summary"]["n1"]["severe_cases"])
    n2_severe = set(report["tail_summary"]["n2"]["severe_cases"])
    report["failure_migration_n1_to_n2"] = {
        "n1_only": sorted(n1_severe - n2_severe),
        "n2_only": sorted(n2_severe - n1_severe),
        "both": sorted(n1_severe & n2_severe),
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path), "csv": str(csv_path), "markdown": str(md_path),
        "three_seed_summary": report["three_seed_summary"],
        "same_dataset_goals": report["same_dataset_goals"],
        "paired_statistics": report["paired_statistics"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
