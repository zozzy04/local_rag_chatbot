"""Script di valutazione quantitativa delle estensioni §5.

Esegue il golden dataset con e senza ogni estensione e calcola il delta
sulle metriche di retrieval (Recall@5, MRR, Hit Rate) e latenza.

Uso:
    python evaluation/evaluate_extensions.py

Output: evaluation/results_extensions.csv + evaluation/results_extensions.png
"""
import json
import time
import logging
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import core_rag_ask
from evaluate import calculate_mrr, calculate_hit_rate, calculate_recall

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_CSV = Path(__file__).parent / "results_extensions.csv"
OUTPUT_PNG = Path(__file__).parent / "bar_extensions_delta.png"
K = 5

CONFIGURATIONS = [
    {"name": "Baseline",        "use_multi_query": False, "use_hybrid_search": False, "use_cache": False, "use_self_correction": False},
    {"name": "Multi-Query",     "use_multi_query": True,  "use_hybrid_search": False, "use_cache": False, "use_self_correction": False},
    {"name": "Hybrid-BM25",     "use_multi_query": False, "use_hybrid_search": True,  "use_cache": False, "use_self_correction": False},
    {"name": "Query-Cache",     "use_multi_query": False, "use_hybrid_search": False, "use_cache": True,  "use_self_correction": False},
    {"name": "Self-Correction", "use_multi_query": False, "use_hybrid_search": False, "use_cache": False, "use_self_correction": True},
]


def run_config(dataset: list, cfg: dict, k: int = 5) -> dict:
    """Esegue il dataset con una configurazione e ritorna le metriche aggregate."""
    mrrs, hits, recalls, latencies = [], [], [], []

    for item in dataset:
        if "Out-of-scope" in item.get("categoria", ""):
            continue

        expected_files = item.get("expected_file")
        if not expected_files:
            continue

        if isinstance(expected_files, str):
            expected_files = [f.strip() for f in expected_files.split(",")]

        t0 = time.time()
        try:
            output = core_rag_ask(
                question=item["domanda"],
                top_k=k,
                use_multi_query=cfg.get("use_multi_query", False),
                use_hybrid_search=cfg.get("use_hybrid_search", False),
                use_cache=cfg.get("use_cache", False),
                use_self_correction=cfg.get("use_self_correction", False),
            )
        except Exception as e:
            logger.error(f"Errore Q{item['id']}: {e}")
            continue
        latency = (time.time() - t0) * 1000

        fonti = output.get("fonti", [])
        mrrs.append(calculate_mrr(fonti, expected_files))
        hits.append(calculate_hit_rate(fonti, expected_files))
        recalls.append(calculate_recall(fonti, expected_files))
        latencies.append(latency)

    n = len(mrrs)
    return {
        "config": cfg["name"],
        "n_questions": n,
        "MRR": round(sum(mrrs) / n, 3) if n else 0,
        "HitRate": round(sum(hits) / n, 3) if n else 0,
        "Recall@k": round(sum(recalls) / n, 3) if n else 0,
        "AvgLatency_ms": round(sum(latencies) / n, 1) if n else 0,
    }


def main():
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for cfg in CONFIGURATIONS:
        logger.info(f"=== Configurazione: {cfg['name']} ===")
        row = run_config(dataset, cfg, k=K)
        results.append(row)
        logger.info(f"  MRR={row['MRR']}  HitRate={row['HitRate']}  Recall@{K}={row['Recall@k']}  Lat={row['AvgLatency_ms']}ms")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info(f"Risultati salvati in {OUTPUT_CSV}")

    # Plot delta rispetto al baseline
    baseline = df[df["config"] == "Baseline"].iloc[0]
    metrics = ["MRR", "HitRate", "Recall@k"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    fig.suptitle("Delta metriche vs Baseline — Estensioni §5", fontsize=14)

    for ax, metric in zip(axes, metrics):
        delta = df[df["config"] != "Baseline"][["config", metric]].copy()
        delta[f"Δ{metric}"] = delta[metric] - float(baseline[metric])
        sns.barplot(data=delta, x="config", y=f"Δ{metric}", ax=ax, palette="coolwarm")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Δ {metric}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    logger.info(f"Grafico salvato in {OUTPUT_PNG}")

    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
