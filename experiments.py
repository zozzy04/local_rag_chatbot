import json
import pandas as pd
import itertools
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

# Importiamo TUTTE le metriche dal tuo evaluate.py
from evaluate import (
    calculate_mrr, 
    calculate_recall, 
    calculate_hit_rate, 
    evaluate_abstention,
    faithfulness,
    answer_correctness
)
from ragas import evaluate as ragas_evaluate
from ragas.run_config import RunConfig
from app.main import ingest_local_documents, core_rag_ask

logging.basicConfig(level=logging.INFO)

# --- DEFINIZIONE MATRICE ---
CHUNK_STRATEGIES = ["SMALL", "MEDIUM", "LARGE"]
EMBEDDING_MODELS = ["nomic-embed-text", "mxbai-embed-large"]
K_VALUES = [3, 5, 10]

# INTERRUTTORE RAGAS: Metti a True se vuoi che Mistral valuti le 540 risposte (richiede molto tempo!)
USE_RAGAS_IN_EXPERIMENTS = True 

def check_citation_accuracy(response: str, sources: list) -> float:
    """Verifica se il LLM ha citato correttamente i file suggeriti."""
    if not sources: return 0.0
    cited_correctly = 0
    for s in sources:
        if s['file'].lower() in response.lower():
            cited_correctly += 1
    return cited_correctly / len(sources)

def run_matrix_experiment():
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results_master = []

    for strategy, emb_model in itertools.product(CHUNK_STRATEGIES, EMBEDDING_MODELS):
        logging.info(f"=== TEST: {strategy} + {emb_model} ===")
        
        # Re-indicizza per questa combinazione
        ingest_local_documents(strategy=strategy, embedding_model_name=emb_model)
        
        for k in K_VALUES:
            logging.info(f"Valutazione k={k}...")
            
            run_metrics = []
            ragas_data = {
                "question": [], "answer": [], "contexts": [], "ground_truth": []
            }
            
            for item in dataset:
                output = core_rag_ask(item['domanda'], top_k=k, embedding_model_name=emb_model)
                risposta = output['risposta_diretta']
                fonti = output['fonti']
                
                # 1. Calcolo di TUTTE le Metriche Manuali
                mrr = calculate_mrr(fonti, item.get('expected_file'))
                recall = calculate_recall(fonti, item.get('expected_file'))
                hit_rate = calculate_hit_rate(fonti, item.get('expected_file'))
                cit_acc = check_citation_accuracy(risposta, fonti)
                abstention_metrics = evaluate_abstention(risposta, item['categoria'])
                
                run_metrics.append({
                    "mrr": mrr,
                    "recall": recall,
                    "hit_rate": hit_rate,
                    "citation_acc": cit_acc,
                    "correct_abstention": abstention_metrics["correct_abstention"],
                    "hallucination": abstention_metrics["hallucination"]
                })
                
                # 2. Preparazione dati per RAGAS (Ignorando astensioni e domande senza ground_truth)
                is_out_of_scope = "Out-of-scope" in item['categoria']
                ground_truth = item.get('ground_truth', "N/A")
                
                if USE_RAGAS_IN_EXPERIMENTS and not is_out_of_scope and ground_truth != "N/A" and "ASTENSIONE:" not in risposta:
                    ragas_data["question"].append(item['domanda'])
                    ragas_data["answer"].append(risposta)
                    ragas_data["contexts"].append([f["estratto"] for f in fonti])
                    ragas_data["ground_truth"].append(ground_truth)
            
            # Calcolo aggregato RAGAS per questa configurazione
            avg_faithfulness = 0.0
            avg_answer_corr = 0.0
            
            if USE_RAGAS_IN_EXPERIMENTS and len(ragas_data["question"]) > 0:
                logging.info("Calcolo RAGAS in corso...")
                ragas_dataset = Dataset.from_dict(ragas_data)
                ragas_res = ragas_evaluate(
                    ragas_dataset, 
                    metrics=[faithfulness, answer_correctness],
                    run_config=RunConfig(timeout=600, max_workers=1),
                    raise_exceptions=False
                )
                df_ragas = ragas_res.to_pandas()
                avg_faithfulness = df_ragas['faithfulness'].mean()
                avg_answer_corr = df_ragas['answer_correctness'].mean()

            # Aggregazione finale nel DataFrame
            df_run = pd.DataFrame(run_metrics)
            
            results_master.append({
                "Strategy": strategy,
                "Embedding": emb_model,
                "K": k,
                "Avg_MRR": df_run['mrr'].mean(),
                "Avg_Recall": df_run['recall'].mean(),
                "Avg_Hit_Rate": df_run['hit_rate'].mean(),
                "Avg_Citation_Acc": df_run['citation_acc'].mean(),
                "Correct_Abstention_Rate": df_run['correct_abstention'].dropna().mean(),
                "Hallucination_Rate": df_run['hallucination'].dropna().mean(),
                "Ragas_Faithfulness": avg_faithfulness,
                "Ragas_Answer_Corr": avg_answer_corr
            })

    # Salva il report definitivo
    master_df = pd.DataFrame(results_master)
    master_df.to_csv("esperimento_confronto_finale.csv", index=False)
    
    # Genera grafici
    generate_charts(master_df)

def generate_charts(df):
    # Generiamo una Heatmap per la Recall e una per l'MRR
    for metric, filename in [('Avg_Recall', 'heatmap_recall.png'), ('Avg_MRR', 'heatmap_mrr.png')]:
        plt.figure(figsize=(10, 6))
        pivot = df.pivot_table(index='Strategy', columns='K', values=metric, aggfunc='mean')
        sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(f"Analisi {metric}: Chunking vs K")
        plt.savefig(filename)
        logging.info(f"Grafico generato: {filename}")

if __name__ == "__main__":
    run_matrix_experiment()