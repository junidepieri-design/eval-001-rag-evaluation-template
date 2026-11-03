"""Step 2: check the input CSV before spending money on LLM calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import eval_config


def parse_contexts(value):
    """Parse the retrieved_contexts cell (a JSON list as text).

    Returns the list of strings, or None if the cell is not a valid
    non-empty JSON list of non-empty strings.
    """
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(items, list) or not items:
        return None
    if not all(isinstance(x, str) and x.strip() for x in items):
        return None
    return items


def validate(df: pd.DataFrame):
    """Return a list of human-readable problems (empty list = valid)."""
    errors: list[str] = []

    missing = [c for c in eval_config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]

    if df.empty:
        return ["The CSV has no rows."]

    for col in ("user_input", "reference", "response"):
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for i in df.index[blank]:
            errors.append(f"Row {i + 1}: '{col}' is empty.")

    for i, value in df["retrieved_contexts"].items():
        if parse_contexts(value) is None:
            errors.append(
                f"Row {i + 1}: 'retrieved_contexts' must be a JSON list of "
                f'non-empty strings, e.g. ["chunk 1", "chunk 2"].'
            )

    dup = df["user_input"].duplicated(keep=False)
    if dup.any():
        rows = [str(i + 1) for i in df.index[dup]]
        errors.append(f"Duplicated questions in rows: {', '.join(rows)}.")

    return errors


def load_and_validate(path: Path | str = eval_config.INPUT_CSV):
    """Read the CSV, validate it, and return it with contexts parsed to lists.

    Raises ValueError listing every problem found.
    """
    df = pd.read_csv(path)
    errors = validate(df)
    if errors:
        raise ValueError("Invalid input CSV:\n- " + "\n- ".join(errors))
    df = df.copy()
    df["retrieved_contexts"] = df["retrieved_contexts"].map(parse_contexts)
    return df


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else eval_config.INPUT_CSV
    try:
        data = load_and_validate(csv_path)
    except ValueError as exc:
        print(exc)
        sys.exit(1)
    print(f"OK: {len(data)} rows, columns {list(data.columns)}")
