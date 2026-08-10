#!/usr/bin/env python3
"""Aggregate formal Stage-1 test60 runs and compare them case by case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl", nargs=3, required=True, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--iga180", required=True)
    parser.add_argument("--iga1800", required=True)
    parser.add_argument("--p2", default="")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    return parser.parse_args()


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_cases(payload: dict, method: str) -> dict[str, dict]:
    record = payload.get("methods", {}).get(method, {})
    if record.get("status") != "completed":
        raise RuntimeError(f"{method} result is not completed")
    cases = record.get("cases", [])
    if len(cases) != 60 or not all(case.get("completed") for case in cases):
        raise RuntimeError(f"{method} must contain 60 completed cases; got {len(cases)}")
    result = {case["case"]: case for case in cases}
    if len(result) != 60:
        raise RuntimeError(f"{method} contains duplicate case names")
    return result


def paired_stats(left: np.ndarray, right: np.ndarray, rng: np.random.Generator, samples: int) -> dict:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    boot = delta[indices].mean(axis=1)
    nonzero = delta[delta != 0.0]
    wilcoxon_p = float(stats.wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
    sample_std = float(np.std(delta, ddof=1)) if len(delta) > 1 else math.nan
    return {
        "n": int(len(delta)),
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wins": int(np.sum(delta < 0.0)),
        "ties": int(np.sum(delta == 0.0)),
        "losses": int(np.sum(delta > 0.0)),
        "cohens_d": float(np.mean(delta) / sample_std) if sample_std > 0.0 else 0.0,
        "t_p": float(stats.ttest_1samp(delta, 0.0).pvalue),
        "wilcoxon_p": wilcoxon_p,
        "gap_gt_800": int(np.sum(delta > 800.0)),
        "lead_gt_800": int(np.sum(delta < -800.0)),
        "max_gap": float(np.max(delta)),
        "max_lead": float(-np.min(delta)),
    }


def method_summary(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def group_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    result = {}
    for name, items in sorted(groups.items()):
        result[name] = {
            "n": len(items),
            "drl_mean": float(np.mean([item["drl_mean"] for item in items])),
            "iga180_mean": float(np.mean([item["iga180"] for item in items])),
            "iga1800_mean": float(np.mean([item["iga1800"] for item in items])),
            "p2_mean": float(np.mean([item["p2"] for item in items])) if items[0]["p2"] is not None else None,
            "drl_minus_iga180": float(np.mean([item["delta_iga180"] for item in items])),
            "drl_minus_iga1800": float(np.mean([item["delta_iga1800"] for item in items])),
            "drl_minus_p2": float(np.mean([item["delta_p2"] for item in items])) if items[0]["p2"] is not None else None,
        }
    return result


def fmt(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    prefix = "+" if signed else ""
    return f"{float(value):{prefix}.{digits}f}"


def markdown_report(report: dict) -> str:
    lines = [
        "# Stage1 R3 三种子 test60 与 IGA 逐案例报告",
        "",
        "Cmax 越低越好。DRL 为 seed1/seed2/seed3 各自预先选定的最佳 validation checkpoint。",
        "",
        "## 总体结果",
        "",
        "| 方法 | 均值 | 标准差 | 中位数 | 最小 | 最大 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("seed1", "R3 seed1"), ("seed2", "R3 seed2"), ("seed3", "R3 seed3"), ("drl_case_mean", "R3 三种子均值策略"), ("p2", "旧 P2"), ("iga180", "IGA-180"), ("iga1800", "IGA-1800")):
        stats_row = report["method_stats"].get(key)
        if stats_row is None:
            continue
        lines.append(
            f"| {label} | {fmt(stats_row['mean'], 2)} | {fmt(stats_row['std'], 2)} | "
            f"{fmt(stats_row['median'], 1)} | {fmt(stats_row['min'], 0)} | {fmt(stats_row['max'], 0)} |"
        )
    formal = report["formal_goal"]
    lines += [
        "",
        f"三个独立 seed 的 test60 均值为 **{formal['three_seed_mean']:.2f} ± {formal['three_seed_sample_std']:.2f}**（seed 间样本标准差）。",
        f"接近 IGA 门槛 8268.53：**{'通过' if formal['near_iga_passed'] else '未通过'}**；达到 IGA-1800 8027.70：**{'通过' if formal['reach_iga_passed'] else '未通过'}**。",
        "",
        "## 配对统计",
        "",
        "| 比较（R3 三种子案例均值 − 基线） | 均值差 | 中位差 | 95% bootstrap CI | 胜/平/负 | gap>800 | p(Wilcoxon) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("aggregate_vs_p2", "旧 P2"), ("aggregate_vs_iga180", "IGA-180"), ("aggregate_vs_iga1800", "IGA-1800")):
        row = report["paired_statistics"].get(key)
        if row is None:
            continue
        lines.append(
            f"| {label} | {fmt(row['mean_delta'], 2, True)} | {fmt(row['median_delta'], 1, True)} | "
            f"[{fmt(row['ci95'][0], 1, True)}, {fmt(row['ci95'][1], 1, True)}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['gap_gt_800']} | {row['wilcoxon_p']:.4g} |"
        )
    for title, key in (("分布", "distribution_summary"), ("Profile", "profile_summary")):
        lines += [
            "",
            f"## {title}结果",
            "",
            f"| {title} | n | R3 | IGA-180 | IGA-1800 | R3−IGA1800 | R3−旧P2 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for name, row in report[key].items():
            lines.append(
                f"| {name} | {row['n']} | {fmt(row['drl_mean'])} | {fmt(row['iga180_mean'])} | "
                f"{fmt(row['iga1800_mean'])} | {fmt(row['drl_minus_iga1800'], 1, True)} | "
                f"{fmt(row['drl_minus_p2'], 1, True)} |"
            )
    stability = report["seed_stability"]
    lines += [
        "",
        "## Seed 稳定性",
        "",
        f"- 三个 seed 全部优于或等于 IGA-1800：{stability['all_three_beat_iga1800']} 个案例。",
        f"- 三个 seed 全部落后 IGA-1800：{stability['all_three_lose_iga1800']} 个案例。",
        f"- 胜负随 seed 改变：{stability['mixed_vs_iga1800']} 个案例。",
        f"- 单案例三 seed 极差：均值 {stability['mean_range']:.1f}，P95 {stability['p95_range']:.1f}，最大 {stability['max_range']:.1f}。",
        "",
        "## 严重失败迁移（相对 IGA-1800，gap>800）",
        "",
        f"- R3：{report['gap_concentration']['r3_severe_count']} 个，贡献净差距的 {report['gap_concentration']['r3_severe_share_of_net'] * 100:.1f}%。",
        f"- 旧 P2：{report['gap_concentration']['p2_severe_count']} 个。",
        f"- R3 修复的旧严重案例：{', '.join(report['failure_migration']['resolved']) or '无'}。",
        f"- R3 新增的严重案例：{', '.join(report['failure_migration']['new']) or '无'}。",
        f"- 仍然严重的案例：{', '.join(report['failure_migration']['persistent']) or '无'}。",
        "",
        "## 60 个案例逐例结果",
        "",
        "| Case | Profile | Dist | S1 | S2 | S3 | R3均值 | IGA180 | IGA1800 | R3−IGA1800 | R3−P2 | 结论 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['case']} | {row['profile']} | {row['distribution']} | {fmt(row['seed1'], 0)} | "
            f"{fmt(row['seed2'], 0)} | {fmt(row['seed3'], 0)} | {fmt(row['drl_mean'])} | "
            f"{fmt(row['iga180'], 0)} | {fmt(row['iga1800'], 0)} | {fmt(row['delta_iga1800'], 1, True)} | "
            f"{fmt(row['delta_p2'], 1, True)} | {row['classification_iga1800']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    drl_payloads = [read_json(path) for path in args.drl]
    iga180_payload = read_json(args.iga180)
    iga1800_payload = read_json(args.iga1800)
    p2_payload = read_json(args.p2) if args.p2 else None

    drl_maps = [completed_cases(payload, "DRL-G") for payload in drl_payloads]
    iga180 = completed_cases(iga180_payload, "IGA")
    iga1800 = completed_cases(iga1800_payload, "IGA")
    p2 = completed_cases(p2_payload, "DRL-G") if p2_payload else None
    case_names = sorted(iga1800)
    expected = set(case_names)
    named_case_maps = [("IGA-180", iga180)] + [
        (f"seed{i + 1}", mapping) for i, mapping in enumerate(drl_maps)
    ]
    for name, mapping in named_case_maps:
        if set(mapping) != expected:
            raise RuntimeError(f"case mismatch for {name}")
    if p2 is not None and set(p2) != expected:
        raise RuntimeError("case mismatch for P2")

    rows = []
    for case in case_names:
        references = [iga180[case], iga1800[case], *(mapping[case] for mapping in drl_maps)]
        hashes = {record.get("case_sha256") for record in references}
        if p2 is not None:
            hashes.add(p2[case].get("case_sha256"))
        if len(hashes) != 1:
            raise RuntimeError(f"case integrity mismatch for {case}: {hashes}")
        seeds = np.asarray([mapping[case]["makespan"] for mapping in drl_maps], dtype=float)
        mean = float(np.mean(seeds))
        i180 = float(iga180[case]["makespan"])
        i1800 = float(iga1800[case]["makespan"])
        p2_value = float(p2[case]["makespan"]) if p2 is not None else None
        delta1800 = mean - i1800
        if delta1800 <= 0:
            classification = "领先/持平"
        elif delta1800 <= 200:
            classification = "接近"
        elif delta1800 <= 800:
            classification = "落后"
        else:
            classification = "严重落后"
        rows.append({
            "case": case,
            "case_id": iga1800[case].get("case_id"),
            "case_sha256": iga1800[case].get("case_sha256"),
            "profile": iga1800[case].get("profile", "unknown"),
            "distribution": iga1800[case].get("distribution", "unknown"),
            "seed1": float(seeds[0]),
            "seed2": float(seeds[1]),
            "seed3": float(seeds[2]),
            "drl_mean": mean,
            "drl_std": float(np.std(seeds, ddof=1)),
            "drl_range": float(np.ptp(seeds)),
            "iga180": i180,
            "iga1800": i1800,
            "p2": p2_value,
            "delta_iga180": mean - i180,
            "delta_iga1800": delta1800,
            "delta_p2": mean - p2_value if p2_value is not None else None,
            "seed_wins_vs_iga1800": int(np.sum(seeds < i1800)),
            "seed_ties_vs_iga1800": int(np.sum(seeds == i1800)),
            "seed_losses_vs_iga1800": int(np.sum(seeds > i1800)),
            "classification_iga1800": classification,
        })

    arrays = {
        f"seed{i + 1}": np.asarray([row[f"seed{i + 1}"] for row in rows])
        for i in range(3)
    }
    arrays["drl_case_mean"] = np.asarray([row["drl_mean"] for row in rows])
    arrays["iga180"] = np.asarray([row["iga180"] for row in rows])
    arrays["iga1800"] = np.asarray([row["iga1800"] for row in rows])
    if p2 is not None:
        arrays["p2"] = np.asarray([row["p2"] for row in rows])

    rng = np.random.default_rng(args.bootstrap_seed)
    paired = {
        "aggregate_vs_iga180": paired_stats(arrays["drl_case_mean"], arrays["iga180"], rng, args.bootstrap_samples),
        "aggregate_vs_iga1800": paired_stats(arrays["drl_case_mean"], arrays["iga1800"], rng, args.bootstrap_samples),
    }
    if p2 is not None:
        paired["aggregate_vs_p2"] = paired_stats(arrays["drl_case_mean"], arrays["p2"], rng, args.bootstrap_samples)
    for i in range(3):
        paired[f"seed{i + 1}_vs_iga180"] = paired_stats(arrays[f"seed{i + 1}"], arrays["iga180"], rng, args.bootstrap_samples)
        paired[f"seed{i + 1}_vs_iga1800"] = paired_stats(arrays[f"seed{i + 1}"], arrays["iga1800"], rng, args.bootstrap_samples)

    seed_means = np.asarray([np.mean(arrays[f"seed{i + 1}"]) for i in range(3)])
    ranges = np.asarray([row["drl_range"] for row in rows])
    r3_severe = {row["case"] for row in rows if row["delta_iga1800"] > 800.0}
    p2_severe = {
        row["case"] for row in rows
        if row["p2"] is not None and row["p2"] - row["iga1800"] > 800.0
    }
    r3_net_gap = float(sum(row["delta_iga1800"] for row in rows))
    r3_severe_gap = float(sum(
        row["delta_iga1800"] for row in rows if row["case"] in r3_severe
    ))
    report = {
        "schema_version": 1,
        "status": "completed",
        "created_unix_time": time.time(),
        "sources": {
            "drl": [str(Path(path).resolve()) for path in args.drl],
            "iga180": str(Path(args.iga180).resolve()),
            "iga1800": str(Path(args.iga1800).resolve()),
            "p2": str(Path(args.p2).resolve()) if args.p2 else None,
        },
        "integrity": {"case_count": 60, "all_case_names_and_hashes_match": True},
        "method_stats": {name: method_summary(values) for name, values in arrays.items()},
        "formal_goal": {
            "three_seed_means": [float(x) for x in seed_means],
            "three_seed_mean": float(np.mean(seed_means)),
            "three_seed_sample_std": float(np.std(seed_means, ddof=1)),
            "near_iga_threshold": 8268.53,
            "near_iga_passed": bool(np.mean(seed_means) <= 8268.53),
            "iga1800_target": 8027.70,
            "reach_iga_passed": bool(np.mean(seed_means) <= 8027.70),
        },
        "paired_statistics": paired,
        "distribution_summary": group_summary(rows, "distribution"),
        "profile_summary": group_summary(rows, "profile"),
        "seed_stability": {
            "all_three_beat_iga1800": sum(row["seed_losses_vs_iga1800"] == 0 for row in rows),
            "all_three_lose_iga1800": sum(row["seed_wins_vs_iga1800"] == 0 and row["seed_ties_vs_iga1800"] == 0 for row in rows),
            "mixed_vs_iga1800": sum(row["seed_wins_vs_iga1800"] > 0 and row["seed_losses_vs_iga1800"] > 0 for row in rows),
            "mean_range": float(np.mean(ranges)),
            "median_range": float(np.median(ranges)),
            "p95_range": float(np.quantile(ranges, 0.95)),
            "max_range": float(np.max(ranges)),
        },
        "gap_concentration": {
            "r3_net_gap_sum": r3_net_gap,
            "r3_positive_gap_sum": float(sum(max(0.0, row["delta_iga1800"]) for row in rows)),
            "r3_severe_count": len(r3_severe),
            "r3_severe_gap_sum": r3_severe_gap,
            "r3_severe_share_of_net": r3_severe_gap / r3_net_gap,
            "p2_severe_count": len(p2_severe),
        },
        "failure_migration": {
            "resolved": sorted(p2_severe - r3_severe),
            "new": sorted(r3_severe - p2_severe),
            "persistent": sorted(r3_severe & p2_severe),
        },
        "worst_vs_iga1800": sorted(rows, key=lambda row: row["delta_iga1800"], reverse=True)[:10],
        "best_vs_iga1800": sorted(rows, key=lambda row: row["delta_iga1800"])[:10],
        "most_seed_sensitive": sorted(rows, key=lambda row: row["drl_range"], reverse=True)[:10],
        "cases": rows,
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
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "formal_goal": report["formal_goal"],
        "aggregate_vs_iga180": paired["aggregate_vs_iga180"],
        "aggregate_vs_iga1800": paired["aggregate_vs_iga1800"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
