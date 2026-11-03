"""Steps 5 and 6: charts and a simple diagnosis of where the RAG fails."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import eval_config

sns.set_theme(style="whitegrid")

LABELS = {
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "factual_correctness": "Factual correctness",
}


def load_scores(path: Path | str = eval_config.SCORES_CSV):
    """Read the scores CSV written by evaluation.py."""
    df = pd.read_csv(path)
    df["retrieved_contexts"] = df["retrieved_contexts"].map(json.loads)
    return df


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------
def diagnose_row(row: pd.Series, threshold: float = eval_config.LOW_SCORE_THRESHOLD):
    """Say where a question went wrong, using only the three scores.

    - low context recall      -> retrieval did not bring what was needed
    - low faithfulness        -> the answer says things the chunks do not support
    - low factual correctness -> the answer does not match the reference
    """
    problems = []
    if row["context_recall"] < threshold:
        problems.append("retrieval")
    if row["faithfulness"] < threshold:
        problems.append("generation (unfaithful)")
    if row["factual_correctness"] < threshold and not problems:
        problems.append("answer incomplete or wrong")
    return " + ".join(problems) if problems else "ok"


def add_diagnosis(df: pd.DataFrame):
    """Return a copy with 'mean_score' and 'diagnosis' columns."""
    out = df.copy()
    out["mean_score"] = out[eval_config.METRICS].mean(axis=1)
    complete = out[eval_config.METRICS].notna().all(axis=1)
    out["diagnosis"] = "not scored"
    out.loc[complete, "diagnosis"] = out[complete].apply(diagnose_row, axis=1)
    return out


def worst_cases(df: pd.DataFrame, n: int = 5):
    """Table of the n lowest-scoring questions with their diagnosis."""
    d = add_diagnosis(df)
    d = d[d["diagnosis"] != "not scored"].copy()
    d["row"] = d.index + 1
    cols = ["row", "user_input", "response", "reference", *eval_config.METRICS, "diagnosis"]
    return d.sort_values("mean_score").head(n)[cols].round(2).reset_index(drop=True)


def diagnosis_counts(df: pd.DataFrame):
    """How many questions fall in each diagnosis."""
    return add_diagnosis(df)["diagnosis"].value_counts()


# --------------------------------------------------------------------------
# Charts (each returns a matplotlib Figure)
# --------------------------------------------------------------------------
def plot_boxplot(df: pd.DataFrame):
    """Distribution of each metric: shows the spread, not just the mean."""
    long = df.melt(value_vars=eval_config.METRICS, var_name="metric", value_name="score").dropna()
    long["metric"] = long["metric"].map(LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=long, x="metric", y="score", ax=ax, color="#cfe0f3", fliersize=0)
    sns.stripplot(data=long, x="metric", y="score", ax=ax, color="#1f4e79", size=5, alpha=0.7)
    ax.axhline(eval_config.LOW_SCORE_THRESHOLD, ls="--", color="gray", lw=1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("Score distribution per metric")
    fig.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame):
    """One row per question, one column per metric; colour is the score."""
    data = df[eval_config.METRICS].copy()
    data.index = [f"{i + 1}. {q[:45]}{'...' if len(q) > 45 else ''}" for i, q in enumerate(df["user_input"])]
    data.columns = [LABELS[c] for c in data.columns]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(data))))
    sns.heatmap(
        data, ax=ax, vmin=0, vmax=1, cmap="RdYlGn", annot=True, fmt=".2f",
        linewidths=0.5, linecolor="white", cbar_kws={"label": "Score"},
    )
    ax.set_title("Scores per question")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_recall_vs_faithfulness(df: pd.DataFrame):
    """Scatter that separates retrieval problems from generation problems."""
    t = eval_config.LOW_SCORE_THRESHOLD
    d = df.dropna(subset=["context_recall", "faithfulness"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axvline(t, ls="--", color="gray", lw=1)
    ax.axhline(t, ls="--", color="gray", lw=1)
    ax.scatter(d["context_recall"], d["faithfulness"], s=70, color="#1f4e79", alpha=0.75)
    for i, r in d.iterrows():
        ax.annotate(str(i + 1), (r["context_recall"], r["faithfulness"]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)

    kw = dict(fontsize=9, color="#555555", ha="center", va="center")
    above, below = t + 0.04, t - 0.04  # labels sit next to the cut-off line
    ax.text(t / 2, above, "retrieval problem", **kw)
    ax.text(t / 2, below, "both failed", **kw)
    ax.text((1 + t) / 2, below, "generation problem", **kw)
    ax.text((1 + t) / 2, above, "healthy", **kw)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("Context recall (retrieval)")
    ax.set_ylabel("Faithfulness (generation)")
    ax.set_title("Where is the failure? (numbers = row in the CSV)")
    fig.tight_layout()
    return fig


def save_all(df: pd.DataFrame, out_dir: Path | str = eval_config.FIGURES_DIR):
    """Save the three charts as PNGs and return their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "boxplot_metrics.png": plot_boxplot,
        "heatmap_questions.png": plot_heatmap,
        "scatter_recall_faithfulness.png": plot_recall_vs_faithfulness,
    }
    paths = []
    for name, fn in charts.items():
        fig = fn(df)
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    scores = load_scores()
    for p in save_all(scores):
        print("Saved:", p)
    print("\nDiagnosis counts:")
    print(diagnosis_counts(scores).to_string())
    print("\nWorst cases:")
    print(worst_cases(scores).to_string(index=False))
