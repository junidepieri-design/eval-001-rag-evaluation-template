'''Central configuration. Edit this file, not the code in src/.'''
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
FIGURES_DIR = ROOT / 'figures'

INPUT_CSV = DATA_DIR / 'eval_input.csv'
SCORES_CSV = DATA_DIR / 'eval_scores.csv'

# Columns the input CSV must have.
REQUIRED_COLUMNS = ['user_input', 'reference', 'response', 'retrieved_contexts']

# Judge LLM used by Ragas to grade. Use a model DIFFERENT from the one
# inside the RAG being evaluated, or the judge tends to grade itself kindly.
JUDGE_PROVIDER = 'openai'  # 'openai' or 'google'
JUDGE_MODELS = {
    'openai': 'gpt-4o-mini',
    'google': 'gemini-2.5-flash',  # check the model is still available for your key
}

# Metrics reported (column names in the scores file).
METRICS = ['context_recall', 'faithfulness', 'factual_correctness']

# A score below this is treated as 'low' in plots and in the diagnosis.
LOW_SCORE_THRESHOLD = 0.7
