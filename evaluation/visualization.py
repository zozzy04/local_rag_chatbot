import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configurazione del logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_charts(df=None):
    """
    Genera i 6 grafici definitivi per l'analisi.
    Se chiamata senza dataframe (es. esecuzione manuale), carica il CSV salvato.
    """
    if df is None:
        logger.info("Caricamento del CSV in corso...")
        df = pd.read_csv("data/results/esperimento_confronto_finale.csv")

    os.makedirs("data/results", exist_ok=True)
    ordine_strategie = ["SMALL", "MEDIUM", "LARGE"]
    
    # 1. HEATMAP SEPARATE PER MODELLO (RECALL E MRR)
    modelli = df['Embedding'].unique()
    metriche = [('Avg_Recall', 'recall'), ('Avg_MRR', 'mrr')]

    for modello in modelli:
        df_modello = df[df['Embedding'] == modello]
        nome_breve = modello.split('-')[0]
        
        for colonna_metrica, nome_metrica in metriche:
            plt.figure(figsize=(7, 5))
            pivot = df_modello.pivot_table(index='Strategy', columns='K', values=colonna_metrica, aggfunc='mean')
            pivot = pivot.reindex(ordine_strategie)
            
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title(f"{colonna_metrica} - Modello: {modello}")
            plt.ylabel("Strategia di Chunking")
            plt.tight_layout()
            
            filepath = f"data/results/heatmap_{nome_metrica}_{nome_breve}.png"
            plt.savefig(filepath)
            plt.close()
            logger.info(f"Heatmap salvata: {filepath}")

    # 2. GRAFICO A BARRE: RAGAS ANSWER CORRECTNESS
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x='Strategy', y='Ragas_Answer_Corr', hue='Embedding', order=ordine_strategie, errorbar=None, palette="viridis")
    plt.title("Qualità della Risposta Finale (RAGAS Answer Correctness)")
    plt.ylabel("Punteggio RAGAS (0-1)")
    plt.xlabel("Strategia di Chunking")
    plt.ylim(0, 1) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout()
    
    filepath_ragas = "data/results/bar_ragas_correctness.png"
    plt.savefig(filepath_ragas)
    plt.close()
    logger.info(f"Grafico RAGAS salvato: {filepath_ragas}")

    # 3. GRAFICO A BARRE: SICUREZZA E GUARDRAILS
    plt.figure(figsize=(6, 5))
    safety_data = {
        'Metrica': ['Astensione Corretta\n(Sicurezza)', 'Allucinazione\n(Rischio)'],
        'Valore': [df['Correct_Abstention_Rate'].mean(), df['Hallucination_Rate'].mean()]
    }
    df_safety = pd.DataFrame(safety_data)
    
    sns.barplot(data=df_safety, x='Metrica', y='Valore', palette=["#2ecc71", "#e74c3c"])
    plt.title("Efficacia dei Guardrail (Domande Out-of-Scope)")
    plt.ylabel("Tasso Medio (0-1)")
    plt.ylim(0, 1.1)
    
    for index, row in df_safety.iterrows():
        plt.text(index, row.Valore + 0.02, f'{row.Valore:.2f}', color='black', ha="center")
        
    plt.tight_layout()
    filepath_safety = "data/results/bar_safety_guardrails.png"
    plt.savefig(filepath_safety)
    plt.close()
    logger.info(f"Grafico Sicurezza salvato: {filepath_safety}")
    logger.info("Tutti e 6 i grafici sono stati generati con successo!")

if __name__ == "__main__":
    generate_charts()