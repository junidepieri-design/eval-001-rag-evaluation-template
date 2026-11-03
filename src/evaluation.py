"""Steps 3 and 4: run Ragas on the validated CSV and save the scores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import FactualCorrectness, Faithfulness, LLMContextRecall

import eval_config
from src.validation import load_and_validate

load_dotenv()

def build_judge():
    """Create the judge LLM from config.JUDGE_PROVIDER (temperature 0)."""
    provider = eval_config.JUDGE_PROVIDER
    model = eval_config.JUDGE_MODELS[provider]

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set (see .env.example).")
        llm = ChatOpenAI(model=model, temperature=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not os.getenv("GOOGLE_API_KEY"):
            raise EnvironmentError("GOOGLE_API_KEY is not set (see .env.example).")
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    else:
        raise ValueError(f"Unknown JUDGE_PROVIDER: {provider!r}")

    return LangchainLLMWrapper(llm)


def to_ragas_dataset(df: pd.DataFrame):
    """Turn the validated DataFrame into a Ragas EvaluationDataset."""
    rows = [
        {
            "user_input": r.user_input,
            "reference": r.reference,
            "response": r.response,
            "retrieved_contexts": list(r.retrieved_contexts),
        }
        for r in df.itertuples(index=False)
    ]
    return EvaluationDataset.from_list(rows)


def _normalize_metric_columns(scores: pd.DataFrame):
    """Ragas may name a column 'factual_correctness(mode=f1)'; keep plain names."""
    rename = {}
    for col in scores.columns:
        for metric in eval_config.METRICS:
            if col != metric and col.startswith(metric):
                rename[col] = metric
    return scores.rename(columns=rename)


def run_evaluation(df: pd.DataFrame, judge: LangchainLLMWrapper | None = None):
    """Run the three metrics and return one row per question with its scores."""
    judge = judge or build_judge()
    result = evaluate(
        dataset=to_ragas_dataset(df),
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=judge,
        raise_exceptions=False,
    )
    scores = _normalize_metric_columns(result.to_pandas())

    out = df.reset_index(drop=True).copy()
    for metric in eval_config.METRICS:
        out[metric] = pd.to_numeric(scores[metric], errors="coerce").values
    return out


def save_scores(scores: pd.DataFrame, path: Path | str = eval_config.SCORES_CSV):
    """Save scores; contexts go back to JSON text so the CSV round-trips."""
    to_save = scores.copy()
    to_save["retrieved_contexts"] = to_save["retrieved_contexts"].map(
        lambda c: json.dumps(c, ensure_ascii=False)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save.to_csv(path, index=False)
    return path


def print_summary(scores: pd.DataFrame):
    """Print mean/median/std per metric and how many scores failed (NaN)."""
    summary = scores[eval_config.METRICS].agg(["mean", "median", "std"]).round(3).T
    summary["failed"] = scores[eval_config.METRICS].isna().sum()
    print("\nSummary (0 to 1, higher is better):")
    print(summary.to_string())
    if summary["failed"].sum():
        print("\nNaN means the judge call failed for that metric (check API key, quota, model name).")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG CSV with Ragas.")
    parser.add_argument("--input", default=eval_config.INPUT_CSV, help="input CSV path")
    parser.add_argument("--output", default=eval_config.SCORES_CSV, help="scores CSV path")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    args = parser.parse_args()

    df = load_and_validate(args.input)
    if args.limit:
        df = df.head(args.limit)

    n = len(df)
    print(
        f"Evaluating {n} rows with judge '{eval_config.JUDGE_MODELS[eval_config.JUDGE_PROVIDER]}' "
        f"({eval_config.JUDGE_PROVIDER}). Expect several LLM calls per row "
        f"(roughly {n * len(eval_config.METRICS) * 3}+ in total)."
    )

    scores = run_evaluation(df)
    path = save_scores(scores, args.output)
    print_summary(scores)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
