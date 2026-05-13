import json
import pandas as pd
from typing import List, Union
import logging
from datasets import Dataset

# Setup RAGAS e Langchain
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_correctness
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Importa le tue funzioni core dal modulo app
try:
    from app.main import core_rag_ask
except ImportError:
    print("Assicurati di lanciare lo script dalla root del progetto (dove si trova app/).")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configurazione Modelli per RAGAS ---
OLLAMA_URL = "http://ollama:11434" 
RAGAS_EVAL_MODEL = "mistral"          
RAGAS_EMBEDDING_MODEL = "nomic-embed-text"

# Inizializza i modelli standard Langchain
llm_judge = ChatOllama(base_url=OLLAMA_URL, model=RAGAS_EVAL_MODEL)
embedder = OllamaEmbeddings(base_url=OLLAMA_URL, model=RAGAS_EMBEDDING_MODEL)

# Incartali nei Wrapper di Ragas
ragas_llm = LangchainLLMWrapper(llm_judge)
ragas_emb = LangchainEmbeddingsWrapper(embedder)

# Assegna i Wrapper alle metriche
faithfulness.llm = ragas_llm
answer_correctness.llm = ragas_llm
answer_correctness.embeddings = ragas_emb

# ==========================================
# 1. METRICHE MANUALI: RETRIEVAL
# ==========================================

def calculate_mrr(retrieved_sources: List[dict], expected_source_files: Union[str, List[str]]) -> float:
    """Calcola il Reciprocal Rank. Prende il primo file utile trovato."""
    if not expected_source_files:
        return 0.0 
    
    if isinstance(expected_source_files, str):
        expected_source_files = [expected_source_files]
        
    for rank, source in enumerate(retrieved_sources, start=1):
        if source.get('file') in expected_source_files:
            return 1.0 / rank # Restituisce il rank del primo match
    return 0.0

def calculate_hit_rate(retrieved_sources: List[dict], expected_source_files: Union[str, List[str]]) -> float:
    """Restituisce 1 se ALMENO UNA fonte attesa è recuperata."""
    if not expected_source_files:
        return 0.0
        
    if isinstance(expected_source_files, str):
        expected_source_files = [expected_source_files]
        
    for source in retrieved_sources:
         if source.get('file') in expected_source_files:
             return 1.0
    return 0.0

def calculate_recall(retrieved_sources: List[dict], expected_source_files: Union[str, List[str]]) -> float:
    """
    Calcola la Recall: (Fonti attese trovate nei chunk) / (Totale fonti attese).
    """
    if not expected_source_files:
        return 0.0
        
    # Assicuriamoci che expected_source_files sia una lista
    if isinstance(expected_source_files, str):
        expected_source_files = [expected_source_files]
        
    totale_attese = len(expected_source_files)
    
    # Crea un set dei file unici recuperati dai chunk
    file_recuperati_unici = set([source.get('file') for source in retrieved_sources])
    
    # Conta quante delle fonti attese sono presenti nel set dei file recuperati
    fonti_trovate = 0
    for file_atteso in expected_source_files:
        if file_atteso in file_recuperati_unici:
            fonti_trovate += 1
            
    return fonti_trovate / totale_attese

# ==========================================
# 2. METRICHE MANUALI: ASTENSIONE
# ==========================================
def evaluate_abstention(response: str, category: str) -> dict:
    """Valuta l'astensione e le allucinazioni per domande Out-of-scope."""
    is_out_of_scope = "Out-of-scope" in category
    has_abstained = "ASTENSIONE:" in response
    
    result = {"correct_abstention": None, "hallucination": None}
    
    if is_out_of_scope:
        if has_abstained:
            result["correct_abstention"] = 1.0
            result["hallucination"] = 0.0
        else:
            result["correct_abstention"] = 0.0
            result["hallucination"] = 1.0
    return result

# ==========================================
# 3. SCRIPT PRINCIPALE
# ==========================================
def run_evaluation(k_value: int = 4):
    logging.info(f"Avvio valutazione con k={k_value}...")
    
    # Carica il Golden Dataset corretto (che ora dovrebbe avere expected_file)
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    risultati_manuali = []
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [] 
    }
    
    for item in dataset:
        logging.info(f"Q{item['id']}: {item['domanda'][:40]}...")
        
        # 1. Esegui la query
        rag_output = core_rag_ask(question=item['domanda'], top_k=k_value)
        risposta = rag_output.get("risposta_diretta", "")
        fonti = rag_output.get("fonti", [])
        
        # 2. Recupera i dati attesi dal dataset
        # Ora gestiamo expected_file in modo che possa essere una stringa o una lista
        expected_files = item.get('expected_file') 
        ground_truth = item.get('ground_truth', "N/A")

        # 3. Calcolo Metriche Manuali di Retrieval
        mrr = calculate_mrr(fonti, expected_files)
        hit_rate = calculate_hit_rate(fonti, expected_files)
        recall = calculate_recall(fonti, expected_files) # Aggiunta Recall!
        
        # 4. Calcolo Metriche Astensione
        abstention_metrics = evaluate_abstention(risposta, item['categoria'])
        
        risultati_manuali.append({
            "id": item['id'],
            "categoria": item['categoria'],
            "MRR": mrr,
            "Hit_Rate": hit_rate,
            "Recall": recall, # Aggiunta al report
            "correct_abstention": abstention_metrics["correct_abstention"],
            "hallucination": abstention_metrics["hallucination"],
            "num_fonti": len(fonti)
        })
        
        # 5. Prepara dati per RAGAS
        # Valutiamo con RAGAS solo se c'è una ground_truth valida e non è un'astensione corretta
        is_out_of_scope = "Out-of-scope" in item['categoria']
        
        if not is_out_of_scope and ground_truth != "N/A":
             ragas_data["question"].append(item['domanda'])
             ragas_data["answer"].append(risposta)
             ragas_data["contexts"].append([f["estratto"] for f in fonti])
             ragas_data["ground_truth"].append(ground_truth)
             
    # ==========================================
    # 4. VALUTAZIONE RAGAS
    # ==========================================
    logging.info("Avvio valutazione generativa con RAGAS...")
    ragas_dataset = Dataset.from_dict(ragas_data)
    
    if len(ragas_dataset) > 0:
        ragas_results = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_correctness]
        )
        df_ragas = ragas_results.to_pandas()
        logging.info("Valutazione RAGAS completata.")
    else:
         logging.warning("Nessun dato valido per RAGAS.")
         df_ragas = pd.DataFrame()

    # ==========================================
    # 5. SALVATAGGIO E REPORT
    # ==========================================
    df_manuale = pd.DataFrame(risultati_manuali)
    df_manuale.to_csv("report_metriche_manuali.csv", index=False)
    if not df_ragas.empty:
        df_ragas.to_csv("report_metriche_ragas.csv", index=False)
        
    logging.info("--- SOMMARIO METRICHE MANUALI ---")
    logging.info(f"MRR Medio: {df_manuale['MRR'].mean():.2f}")
    logging.info(f"Hit Rate Medio: {df_manuale['Hit_Rate'].mean():.2f}")
    logging.info(f"Recall Media: {df_manuale['Recall'].mean():.2f}") # Stampa della Recall media
    
    correct_abstention = df_manuale['correct_abstention'].dropna().mean()
    hallucination_rate = df_manuale['hallucination'].dropna().mean()
    logging.info(f"Correct Abstention Rate (Out-of-scope): {correct_abstention:.2f}")
    logging.info(f"Hallucination Rate (Out-of-scope): {hallucination_rate:.2f}")

    if not df_ragas.empty:
        logging.info("\n--- SOMMARIO METRICHE RAGAS ---")
        logging.info(f"Faithfulness Media: {df_ragas['faithfulness'].mean():.2f}")
        logging.info(f"Answer Correctness Media: {df_ragas['answer_correctness'].mean():.2f}")

if __name__ == "__main__":
    run_evaluation()