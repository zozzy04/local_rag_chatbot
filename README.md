# Sistema RAG su PDF — Progetto GenAI

Sistema **Retrieval-Augmented Generation** offline che indicizza PDF aziendali e risponde a domande in linguaggio naturale citando le fonti, con persistenza relazionale completa e interoperabilità federata tra knowledge base di gruppi distinti.

**Caso d'uso:** Studio professionale IT/Cloud che vuole interrogare manuali tecnici (AI Agents, AI as a Service, AI for Everyday IT) in modo tracciabile, auditabile e federato con i sistemi di altri team.

---

## Architettura

```
┌──────────────────────────────────────────────────────────┐
│  CLI (app.py)                                            │
│  index | ask | serve | db init/history/export/analyze   │
└────────────┬─────────────────────────────────────────────┘
             │
    ┌────────▼────────┐          ┌─────────────────────┐
    │  app/main.py    │          │  api/server.py       │
    │  core_rag_ask() │◄────────►│  FastAPI (spec v1.0) │
    │  ingest_docs()  │          │  /health /info       │
    └───┬────────┬────┘          │  /documents          │
        │        │               │  /retrieve /ask      │
   ┌────▼──┐  ┌──▼──────────┐   └──────────┬───────────┘
   │Qdrant │  │ SQLAlchemy  │              │
   │Vector │  │ SQLite DB   │    ┌──────────▼──────────┐
   │Store  │  │ (Alembic)   │    │  api/federation.py   │
   └───────┘  └─────────────┘    │  RRF peer retrieval  │
                                 └─────────────────────┘
    ┌─────────────────────────────────┐
    │  app/extensions/               │
    │  multi_query | hybrid_search   │
    │  query_cache | self_correction │
    │  structured_logger             │
    └─────────────────────────────────┘
```

---

## Setup rapido

### 1. Variabili d'ambiente

```bash
cp .env.example .env
# Configura: GROUP_ID, GROUP_NAME, API_KEY, DATABASE_URL
```

### 2. Dipendenze Python

```bash
pip install -r requirements.txt
pip install alembic slowapi rank_bm25
```

### 3. Avvio servizi (Docker)

```bash
docker-compose up -d   # avvia Ollama + Qdrant
```

### 4. Inizializza il DB

```bash
python app.py db init
# Esegue le migrazioni Alembic + crea il record del gruppo locale
```

### 5. Indicizza i PDF

```bash
# Metti i PDF in data/input/
python app.py index               # preset MEDIUM (default)
python app.py index --preset LARGE
```

---

## Comandi CLI

### Interrogazione locale

```bash
python app.py ask "Cosa sono gli agenti AI?"
python app.py ask "Cos'è AWS Polly?" --source AI_as_a_Service.pdf
python app.py ask "Spiega i token" -k 8
```

### Con estensioni avanzate (paragrafo 5)

```bash
python app.py ask "Domanda..." --multi-query      # Multi-query expansion
python app.py ask "Domanda..." --hybrid           # Hybrid BM25+dense search
python app.py ask "Domanda..." --cache            # Semantic query cache
python app.py ask "Domanda..." --self-correct     # Self-correction loop
python app.py ask "Domanda..." --multi-query --hybrid --self-correct
```

### Interrogazione federata (paragrafo 7)

```bash
python app.py ask "Domanda cross-group" --federated
python app.py ask "Domanda..." --remote groupB,groupC
```

### Server API federata

```bash
python app.py serve               # http://0.0.0.0:8000
python app.py serve --port 8001   # porta custom
# Docs interattive: http://localhost:8000/docs
```

### Gestione DB (paragrafo 6)

```bash
python app.py db init
python app.py db history --last 50
python app.py db export --format csv
python app.py db analyze          # 3 query analitiche paragrafo 6.2
```

---

## API REST (paragrafo 7.2)

Header obbligatori (eccetto `/health`):

```
X-API-Key: <valore da .env>
X-API-Version: 1.0
X-Group-Id: <id del gruppo chiamante>
```

| Metodo | Endpoint     | Descrizione                                      |
|--------|--------------|--------------------------------------------------|
| GET    | `/health`    | Liveness check (no auth, < 100ms)               |
| GET    | `/info`      | Metadata KB: n_docs, n_chunks, modello embedding |
| GET    | `/documents` | Lista PDF indicizzati                            |
| POST   | `/retrieve`  | Top-k chunks senza LLM (timeout 5s)             |
| POST   | `/ask`       | Risposta completa con citazioni (timeout 30s)    |

Esempio:
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "X-API-Key: changeme" -H "X-API-Version: 1.0" -H "X-Group-Id: groupB" \
  -H "Content-Type: application/json" \
  -d '{"question": "Cos'\''e un agente AI?", "k": 5}'
```

---

## Federazione multi-KB (paragrafo 7)

### Configurazione

```yaml
# peers.yaml
peers:
  - id: groupB
    name: "Team Altro"
    base_url: "http://<ip-groupB>:8001"
    api_key: "chiave-condivisa"
    enabled: true
```

### Avvio

```bash
python app.py serve --port 8000   # espone la KB locale
python app.py ask "Domanda" --federated
```

### Strategia di fusion: Reciprocal Rank Fusion (RRF)

RRF combina i ranking di gruppi con embedding diversi senza richiedere che gli score siano comparabili. Formula: `score(d) = sum(1 / (k + rank(d, group)))` per k=60. Robusto alle differenze di scala tra modelli di embedding distinti.

### Resilienza

- Timeout configurabile: `FEDERATION_TIMEOUT_S=5` in `.env`
- Peer offline: la query prosegue con KB disponibili + warning nella risposta
- Ogni risposta federata salva `remote_group_id` nel DB per auditabilità

---

## Estensioni implementate (paragrafo 5)

| # | Estensione | Flag CLI | Impatto atteso |
|---|------------|----------|----------------|
| 2 | **Hybrid Search** BM25+dense+RRF | `--hybrid` | +Recall su termini tecnici esatti |
| 3 | **Multi-Query Expansion** | `--multi-query` | +Recall su query ambigue |
| 7 | **Structured Logging** JSON | sempre attivo | Osservabilità e auditabilita completa |
| 8 | **Query Cache** semantica (cosine threshold 0.95) | `--cache` | -Latenza su query ripetute |
| 9 | **Self-Correction Loop** | `--self-correct` | +Faithfulness |

Valutazione delta metriche:
```bash
python evaluation/evaluate_extensions.py
# Output: evaluation/results_extensions.csv + bar_extensions_delta.png
```

---

## Scelte tecniche motivate

**Vector Store: Qdrant (non Chroma)**
Qdrant è stato scelto per la sua architettura a microservizi Docker-ready, il supporto a payload filtering (usato per `source_filter` e `file_hash`), le performance di produzione e l'API Python stabile. Offre le stesse garanzie di Chroma con maggiore scalabilita orizzontale.

**Embedding: nomic-embed-text (768d) vs mxbai-embed-large (1024d)**
Entrambi testati sperimentalmente. Nomic e piu veloce e compatto; mxbai ha performance superiori su testi tecnici in inglese. La configurazione vincente emerge dall'esperimento di confronto (vedere `evaluation/esperimento_confronto_finale.csv`).

**Chunking: RecursiveCharacterTextSplitter**
Configurabile via `app/config.yaml`. MEDIUM (1000/200) e il preset vincente nel confronto, bilanciando contesto sufficiente e precisione di citazione.

**DB: SQLite + SQLAlchemy 2.0 + Alembic**
SQLite per semplicita di deploy (zero server aggiuntivi). Alembic per migrazioni versionabili. Lo schema relazionale supporta auditabilita completa: chi ha chiesto cosa, quando, con quali fonti, con quale latenza.

---

## Valutazione (paragrafo 4)

```bash
cd evaluation

# Baseline
python evaluate.py

# Matrice chunking x embedding x k
python experiments.py

# Grafici
python visualization.py

# Delta estensioni paragrafo 5
python evaluate_extensions.py
```

Golden dataset: `evaluation/golden_dataset.json` — 35 domande: 8 fattuali singola, 6 multi-chunk, 4 multi-hop, 3 sintesi, 5 out-of-scope, 2 ambigue, 2 edge case, **5 cross-group federati**.

**Risultati baseline (MEDIUM + nomic-embed-text + k=4):**

| Metrica | Valore |
|--------|--------|
| MRR | 0.525 |
| HitRate@4 | 0.733 |
| Recall@4 | 0.733 |
| Astensioni corrette (out-of-scope) | 5/5 — 100% |

---

## Test

```bash
python -m pytest tests/ -v
# 39 test: test_metrics.py | test_db.py | test_api.py
```

---

## Struttura progetto

```
├── app.py                        # CLI entry point
├── app/
│   ├── main.py                   # Core RAG: ingest + ask + estensioni
│   ├── prompt.py                 # Prompt + guardrail
│   ├── config.yaml               # Chunking presets SMALL/MEDIUM/LARGE
│   └── extensions/               # Estensioni paragrafo 5
│       ├── multi_query.py        # #3 Multi-query expansion
│       ├── hybrid_search.py      # #2 BM25+dense+RRF
│       ├── query_cache.py        # #8 Semantic cache
│       ├── structured_logger.py  # #7 JSON logging
│       └── self_correction.py    # #9 Self-correction
├── api/
│   ├── server.py                 # FastAPI conforme a api_spec.yaml
│   └── federation.py             # RRF federated retrieval
├── db/
│   ├── models.py                 # SQLAlchemy ORM (9 tabelle)
│   ├── database.py               # Engine + session context manager
│   ├── analytics.py              # 3 query analitiche paragrafo 6.2
│   └── migrations/versions/001_initial_schema.py
├── evaluation/
│   ├── golden_dataset.json       # 35 domande (5 cross-group)
│   ├── evaluate.py               # Metriche baseline
│   ├── evaluate_extensions.py    # Delta metriche estensioni
│   ├── experiments.py            # Matrice esperimenti
│   └── visualization.py
├── tests/
│   ├── test_metrics.py           # Recall@k, MRR, HitRate, RRF
│   ├── test_db.py                # CRUD + cascade + transazioni
│   └── test_api.py               # Tutti gli endpoint + federation
├── data/input/                   # PDF sorgente
├── logs/rag_logs.jsonl           # Structured logging
├── peers.yaml                    # Peer federation
├── alembic.ini                   # Alembic
└── .env.example                  # Template env
```

---

## Formato risposta (paragrafo 3.5)

```
============================================================
[RISPOSTA]
Il testo della risposta diretta basata solo sui documenti.

[FONTI]
- file: AI_Agents_in_Action.pdf, pag: 42, score: 0.891
  estratto: "I cinque componenti principali di un agente..."

[CONFIDENZA] Alta (Forte corrispondenza)
[MOTIVAZIONE] Punteggio medio vettoriale: 0.86.
============================================================
```

Se l'informazione non e nei documenti:
```
[RISPOSTA]
ASTENSIONE: Le informazioni presenti nei documenti non sono sufficienti per rispondere.
[CONFIDENZA] Bassa (Astensione forzata)
```
