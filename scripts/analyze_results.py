import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "model_outputs"
ANALYSIS_DIR = ROOT / "results" / "analysis"
PLOTS_DIR = ROOT / "results" / "plots"


def load_latest_run() -> Tuple[str, pd.DataFrame]:
    files = sorted(OUTPUT_DIR.glob("*__*.jsonl"))
    if not files:
        raise RuntimeError("No result files found in results/model_outputs.")

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        timestamp = path.name.split("__", 1)[0]
        grouped[timestamp].append(path)

    latest_run_id = sorted(grouped.keys())[-1]
    records: List[Dict] = []
    for path in grouped[latest_run_id]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    return latest_run_id, df


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    margin = (z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_test(df_direct: pd.Series, df_cot: pd.Series) -> Dict[str, float]:
    comparison = pd.DataFrame({"direct": df_direct, "cot": df_cot})
    comparison = comparison.dropna()
    if comparison.empty:
        return {"b": 0, "c": 0, "p_value": float("nan")}
    b = int(((comparison["direct"] == 1) & (comparison["cot"] == 0)).sum())
    c = int(((comparison["direct"] == 0) & (comparison["cot"] == 1)).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"b": b, "c": c, "p_value": p_value}


def cohen_h(p1: float, p2: float) -> float:
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def main() -> None:
    run_id, df = load_latest_run()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df["is_correct"] = df["is_correct"].astype(bool)
    df["correct_numeric"] = df["is_correct"].astype(int)

    summary_rows = []
    domain_rows = []
    for (model, style), group in df.groupby(["model_name", "prompt_style"]):
        n = len(group)
        successes = int(group["correct_numeric"].sum())
        accuracy = successes / n if n else 0.0
        ci_low, ci_high = wilson_interval(successes, n)
        total_prompt_tokens = int(
            group["usage"].apply(lambda x: (x or {}).get("prompt_tokens") or (x or {}).get("input_tokens") or 0).sum()
        )
        total_completion_tokens = int(
            group["usage"]
            .apply(lambda x: (x or {}).get("completion_tokens") or (x or {}).get("output_tokens") or 0)
            .sum()
        )
        summary_rows.append(
            {
                "run_id": run_id,
                "model": model,
                "prompt_style": style,
                "n_examples": n,
                "n_correct": successes,
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }
        )
        for domain, domain_group in group.groupby("domain"):
            dn = len(domain_group)
            dsuccess = int(domain_group["correct_numeric"].sum())
            domain_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "prompt_style": style,
                    "domain": domain,
                    "n_examples": dn,
                    "n_correct": dsuccess,
                    "accuracy": dsuccess / dn if dn else 0.0,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    domain_df = pd.DataFrame(domain_rows)
    summary_df.to_csv(ANALYSIS_DIR / "summary.csv", index=False)
    domain_df.to_csv(ANALYSIS_DIR / "domain_breakdown.csv", index=False)

    comparisons = []
    for model, group in df.groupby("model_name"):
        direct = (
            group[group["prompt_style"] == "direct"]
            .set_index("scenario_id")["correct_numeric"]
        )
        cot = (
            group[group["prompt_style"] == "cot"]
            .set_index("scenario_id")["correct_numeric"]
        )
        aligned = direct.to_frame("direct").join(cot.to_frame("cot"), how="inner")
        if aligned.empty:
            continue
        stats = mcnemar_test(aligned["direct"], aligned["cot"])
        p_direct = aligned["direct"].mean()
        p_cot = aligned["cot"].mean()
        effect = cohen_h(p_cot, p_direct)
        comparisons.append(
            {
                "run_id": run_id,
                "model": model,
                "direct_accuracy": p_direct,
                "cot_accuracy": p_cot,
                "delta": p_cot - p_direct,
                "mcnemar_b": stats["b"],
                "mcnemar_c": stats["c"],
                "mcnemar_p_value": stats["p_value"],
                "cohen_h_cot_vs_direct": effect,
            }
        )

    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(ANALYSIS_DIR / "prompt_comparisons.csv", index=False)

    # Plots
    pivot = summary_df.pivot(index="model", columns="prompt_style", values="accuracy")
    ax = pivot.plot(kind="bar", rot=0, figsize=(10, 6))
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"Conditional Forgetting Accuracy by Model & Prompt (Run {run_id})")
    plt.axhline(1.0, color="black", linewidth=0.5)
    plt.legend(title="Prompt Style")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"accuracy_{run_id}.png", dpi=200)
    plt.close()

    domain_plot = domain_df.pivot_table(
        index=["model", "prompt_style"],
        columns="domain",
        values="accuracy",
    )
    domain_plot.plot(kind="bar", figsize=(12, 7))
    plt.ylabel("Accuracy")
    plt.title(f"Domain Accuracy Breakdown (Run {run_id})")
    plt.ylim(0, 1)
    plt.legend(title="Domain")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"domain_accuracy_{run_id}.png", dpi=200)
    plt.close()

    # Save metadata for report
    (ANALYSIS_DIR / "run_id.txt").write_text(run_id, encoding="utf-8")

    # Export representative failures
    failures = df[~df["is_correct"]].copy()
    failures = failures.sort_values(["model_name", "prompt_style", "domain", "scenario_id"])
    sample_failures = failures.groupby(["model_name", "prompt_style", "domain"]).head(2)
    sample_failures.to_json(ANALYSIS_DIR / "sample_failures.json", orient="records", indent=2)


if __name__ == "__main__":
    main()

