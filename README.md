# Sistema RAG Offline Completo (FastAPI + LangChain + Ollama + Qdrant)

Questo repository contiene un'implementazione avanzata di un sistema RAG (Retrieval-Augmented Generation) completamente offline. Il progetto sfrutta **Ollama** per l'esecuzione locale di Large Language Models (LLMs) ed Embeddings, **Qdrant** come vector database e **FastAPI** come backend server, uniti dalla logica orchestrata da **LangChain**.

Il sistema è progettato per garantire robustezza, riducendo le allucinazioni tramite *guardrails* molto rigidi, consentendo la valutazione sistematica con framework come *RAGAS* e fornendo interfacce sia tramite CLI (Command Line Interface) che tramite API REST.

---

## 🌟 Architettura e Funzionalità Principali

### 1. Ingestion Intelligente
- **Chunking Configurabile**: I documenti in ingresso (PDF) vengono suddivisi in *chunk* tramite parametri configurabili (`app/config.yaml`). Sono disponibili tre preset: `SMALL`, `MEDIUM`, `LARGE` per testare la granularità.
- **Caching basato su Hash (SHA-256)**: Il sistema calcola l'hash dei file in fase di ingestion per evitare la ri-elaborazione degli stessi file già presenti in Qdrant, ottimizzando i tempi di caricamento.
- **Metadati Ricchi**: Ad ogni *chunk* vengono associati nome del file, numero di pagina e hash.

### 2. Recupero e Generazione (RAG Engine)
- **Top-K Dinamico e Filtraggio**: Possibilità di definire quanti frammenti recuperare (`-k`) e l'opzione di limitare la ricerca a un singolo documento target (`--source`).
- **Guardrails e Prompting Rigoroso**: Prompt progettati per:
  - **Risposte Closed-Book**: Basare le risposte ESCLUSIVAMENTE sui documenti.
  - **Citazioni Obbligatorie**: Riportare la fonte (file e pagina) all'interno del testo generato.
  - **Astensione Controllata**: Se il contesto non supporta la domanda (out-of-scope), il sistema deve restituire una stringa esatta di "ASTENSIONE: ...".
- **Self-Correction & Query Rewriting**: Se il modello si astiene, un layer intermedio esegue un *query rewriting* per riformulare la domanda iniziale dell'utente (usando sinonimi o generalizzazioni) e tenta un secondo passaggio di retrieval per aumentare la probabilità di successo.

### 3. API REST e Lifespan FastAPI
- Server asincrono che inizializza la base di dati all'avvio garantendo l'accessibilità immediata delle risorse.
- **Endpoint `/ask`**: Per fare domande (con parametri per k e filtro sorgente).
- **Endpoint `/reindex`**: Lancia l'indicizzazione in un *background task* asincrono, evitando blocchi per gli altri client web connessi.

### 4. Framework di Valutazione (Evaluation & Experiments)
Il sistema prevede script dedicati alla misurazione dell'accuratezza tramite un *golden dataset* di test (`evaluation/golden_dataset.json`).
- **`evaluate.py`**:
  - Calcola *metriche manuali di retrieval*: MRR (Mean Reciprocal Rank), Hit Rate, e Recall.
  - Valuta *metriche di comportamento*: Correct Abstention Rate (per domande out-of-scope) e Hallucination Rate.
  - Integra **RAGAS** per misurare le performance puramente LLM (Faithfulness, Answer Correctness) usando il modello locale come giudice.
- **`experiments.py`**:
  - Esegue una matrice di test esaustiva confrontando:
    - **Strategie di Chunking** (SMALL, MEDIUM, LARGE)
    - **Modelli di Embedding** (es. *nomic-embed-text*, *mxbai-embed-large*)
    - **Valori di Top-K** (3, 5, 10)
  - Mantiene gli esperimenti in collezioni Qdrant temporanee/separate, isolate da quella di produzione.
  - Esporta i risultati globali in CSV (`data/results/`) e genera Heatmaps grafiche per Recall e MRR per giustificare la scelta dei parametri migliori.

---

## 🛠 Setup e Configurazione

1. Clonare il repository.
2. Predisporre i modelli base di **Ollama** che si intendono utilizzare (default: `mistral` per la generazione, `nomic-embed-text` per gli embeddings).
3. Configurare le variabili d'ambiente nel file `.env` (si può prendere come base `.env.example`).
4. Installare le dipendenze Python tramite `pip install -r requirements.txt` (o usare Docker Compose).

Parametri di base `.env`:
```env
OLLAMA_HOST=http://localhost:11434
QDRANT_HOST=http://localhost:6333
OLLAMA_MODEL=mistral
EMBEDDING_MODEL=nomic-embed-text
COLLECTION_NAME=default_collection
SOURCE_DIR=data/input
```

*(Nota: in un contesto Dockerizzato tramite `docker-compose`, i nomi degli host corrisponderanno ai nomi dei servizi, ad esempio `http://ollama:11434` e `http://qdrant:6333`).*

---

## 💻 Interfaccia a Linea di Comando (CLI)

Il file `app.py` funge da punto d'ingresso principale in ambiente CLI.

### Indicizzare i Documenti (Ingestion)
I documenti inseriti nella cartella `data/input` (o quella configurata in `.env`) verranno passati nel database vettoriale.

```bash
# Usa il preset di chunking di default (MEDIUM)
python app.py index

# Usa un preset di chunking specifico
python app.py index --preset SMALL
python app.py index --preset LARGE
```

### Interrogare il RAG (Ask)
Lancia l'interrogazione completa sulla base di dati indicizzata.

```bash
# Domanda semplice (Top-K default = 4)
python app.py ask "Quali sono i concetti chiave dell'AI?"

# Domanda specificando Top-K a 10
python app.py ask "Quali sono i concetti chiave dell'AI?" -k 10

# Domanda filtrata su un documento specifico
python app.py ask "Cosa si dice di X?" --source "Documento_Target.pdf"
```
Il terminale risponderà con una formattazione chiara divisa in `[RISPOSTA]`, `[FONTI]`, `[CONFIDENZA]` e `[MOTIVAZIONE]`.

---

## 🌐 Utilizzo API REST (FastAPI)

Avviando il server FastAPI (es. tramite uvicorn o Docker Compose), l'app espone un server che permette l'integrazione con eventuali frontend o microservizi.

**1. Risposta del RAG (`POST /ask`)**
Parametri Form Data:
- `question` (string): La domanda da porre.
- `k` (int, opzionale, default 4): Quanti chunk recuperare.
- `source` (string, opzionale): Filtro per limitare a un singolo documento target.

**2. Re-indicizzazione asincrona (`POST /reindex`)**
Non accetta parametri. Avvia l'ingestion in un thread secondario, scansionando nuovamente la directory `SOURCE_DIR` per nuovi PDF non ancora indicizzati (grazie al controllo su hash).

---

## 📊 Sistema di Valutazione (Evaluation)

Per testare e provare statisticamente le performance del sistema RAG, spostarsi nella root e lanciare gli script dedicati all'interno di `evaluation/`. I test si basano su un set di query in `evaluation/golden_dataset.json`.

**1. Valutazione standard (`evaluate.py`)**
Effettua un run con i parametri attuali (k=4) sul dataset e genera report CSV (`report_metriche_manuali.csv` e `report_metriche_ragas.csv`). Mostra a schermo statistiche globali su MRR, Hit Rate, Recall e Astensioni corrette.
```bash
python evaluation/evaluate.py
```

**2. Matrice degli Esperimenti (`experiments.py`)**
Esegue tutte le permutazioni dei parametri configurabili per trovare l'impostazione RAG "ottimale" (Hyperparameter tuning). Questa operazione è onerosa a livello computazionale (richiederà svariati minuti/ore in base all'hardware).
```bash
python evaluation/experiments.py
```
I risultati finali (comprese le *Heatmap* visuali per un confronto immediato di MRR e Recall al variare del parametro *K* e della strategia di *Chunking*) verranno salvati nella cartella `data/results/`.

---

## 🔒 Sicurezza e Robustezza

Il sistema prevede un approccio di tipo "**Strict Grounding**".
Il prompt (visibile in `app/prompt.py`) forza severamente le limitazioni, ma l'intero layer logico in `core_rag_ask` supervisiona la coerenza dell'output:
- Se il database vettoriale restituisce un livello di confidenza basso, un sistema euristico lo classifica e avverte l'utente.
- Se la risposta del modello non supera i test di pertinenza e include "ASTENSIONE:", subentra un *fall-back* sicuro con astensione rigida limitando allucinazioni (es. risposte a domande *out-of-scope* non inerenti ai documenti testati).
- Un sistema di self-correction (tramite **query rewriting** basato su LLM) cerca di mitigare il problema se la prima query vettoriale era malformata o troppo stringente, dando un *seconda chance* tramite sinonimizzazione prima di arrendersi all'astensione globale.