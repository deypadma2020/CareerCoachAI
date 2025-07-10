import os
import json
from datetime import datetime

LOG_FILE = "context_classification_log.jsonl"

def ensure_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("")  # create empty file

def log_context_classification(query, label, source, score=None, response_type=None):
    ensure_log_file()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "classification": label,     # 'in_context', 'out_of_context', or 'uncertain'
        "source": source,            # e.g., 'cache_exact', 'cache_fuzzy', 'llm_classification'
        "score": round(score, 4) if score is not None else None,
        "response_type": response_type  # e.g., 'memory_exact', 'memory_fuzzy', 'llm_generated'
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_all_logs():
    ensure_log_file()
    with open(LOG_FILE, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def clear_logs():
    with open(LOG_FILE, "w") as f:
        f.write("")
