# Documentazione di Sistema: RAG Backend (FastAPI + LangChain)
Questo documento illustra l'architettura, la configurazione e il funzionamento del backend per il sistema RAG (Retrieval-Augmented Generation) locale.

L'infrastruttura permette di interrogare in linguaggio naturale una base di conoscenza privata (file PDF), sfruttando Ollama come motore di Intelligenza Artificiale e Qdrant come database vettoriale.

1. **Configurazione e Logging**
La fase di setup garantisce che l'applicazione sia disaccoppiata dall'infrastruttura sottostante tramite variabili d'ambiente e che ogni azione sia tracciata.

2. **Gestione Variabili d'Ambiente (.env)**
Per evitare l'hardcoding, i parametri vitali vengono estratti tramite dotenv. Se una variabile manca, il sistema adotta un valore di fallback sicuro

#### Indirizzi dei servizi e modelli AI
```
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
```
#### Configurazione Dati
```
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")
SOURCE_DIR = os.getenv("SOURCE_DIR", "/app/documenti_da_indicizzare")
```
#### Strategie di Chunking
Tagliare i documenti in frammenti (chunk) è vitale nel RAG. Il sistema prevede tre profili pronti all'uso:

**size**: Il numero massimo di caratteri per frammento.

**overlap**: La sovrapposizione tra un frammento e l'altro per mantenere il contesto semantico.

```
CHUNK_PRESETS = {
    "SMALL": {"size": 400, "overlap": 80},     # Per dettagli granulari
    "MEDIUM": {"size": 1000, "overlap": 200},  # Bilanciamento standard
    "LARGE": {"size": 2000, "overlap": 300}    # Per contesti molto ampi
}
```

#### Logging Strutturato
Sostituisce i comandi print con il modulo nativo logging in formato Syslog, indispensabile per il monitoraggio in produzione:

```
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
```
# Il Motore RAG: Indicizzazione Dati
La funzione ingest_local_documents() gestisce l'intero processo di lettura e salvataggio dei PDF nel database vettoriale, diviso in fasi distinte.

### Fase A: Gestione Cache
Per ridurre i tempi di avvio e lo sforzo della GPU, il sistema verifica se i dati sono già presenti in Qdrant:

```
collections = client.get_collections().collections
exists = any(c.name == COLLECTION_NAME for c in collections)

if exists:
    logger.info("CACHE TROVATA. Salto l'ingestion per risparmiare risorse.")
    return # Evita di rileggere i PDF
```
### Fase B: Estrazione e Chunking
Se la cache è vuota, il sistema carica i PDF e li scompone intelligentemente:

1. Estrazione del testo dal PDF
```
loader = PyPDFLoader(pdf_path)
pages = loader.load()
```
2. Suddivisione semantica (Chunking)
```
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["size"],
    chunk_overlap=config["overlap"],
    separators=["\n\n", "\n", ".", " "] # Ordine di preferenza per i tagli
)
chunks = text_splitter.split_documents(pages)
```
### Fase C: Embedding e Storage
I frammenti di testo vengono inviati a Ollama per essere convertiti in coordinate matematiche (vettori) e infine salvati in Qdrant:

```
vector_store = QdrantVectorStore.from_documents(
    documents=all_chunks,
    embedding=embeddings, # Modello nomic-embed-text
    url=QDRANT_HOST,
    collection_name=COLLECTION_NAME,
    force_recreate=True 
)
```
#### Ciclo di Vita (Lifespan)
Il server utilizza il costrutto moderno **@asynccontextmanager** per assicurarsi che la base di conoscenza sia pronta prima di accettare richieste web esterne:

```
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inizializzazione del sistema RAG...")
    ingest_local_documents() # Popola o carica la memoria
    yield                    # Il server inizia ad accettare chiamate HTTP
    logger.info("Chiusura del server in corso. Pulizia risorse...")

app = FastAPI(lifespan=lifespan)
```
## Gli Endpoint API
Il livello esposto agli utenti è gestito da FastAPI. Entrambi gli endpoint accettano richieste in POST.

#### Endpoint 1: /reindex
Permette di far leggere al sistema nuovi PDF inseriti nella cartella, delegando il calcolo pesante a un Background Task, senza bloccare il traffico in entrata:

```
@app.post("/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    logger.info("Avvio re-indicizzazione in background...")
    background_tasks.add_task(ingest_local_documents)
    return {"status": "ok", "dettaglio": "Elaborazione avviata in background."}
```
#### Endpoint 2: /ask (La Pipeline di Risposta)
Questo endpoint orchestra l'intera comunicazione tra l'utente, il database vettoriale e il modello linguistico locale (Mistral).

1. Retrieval (Recupero Semantico)
Il sistema converte la domanda in vettore e recupera i k=4 frammenti di testo più simili dal database:

```
results_with_score = vs.similarity_search_with_score(question, k=4)
context = "\n".join([doc.page_content for doc, _ in results_with_score])
```

2. Prompting
Per evitare che l'AI "allucini" o inventi risposte non presenti nei documenti, viene imposto un PromptTemplate rigoroso:

```
prompt = PromptTemplate.from_template("""
Rispondi in Italiano basandoti SOLO sul contesto fornito. 
Se le informazioni non sono sufficienti, dillo chiaramente.
Contesto: {context}
Domanda: {question}
Risposta:""")
```

3. Generation (Invocazione dell'LLM)
Il contesto fuso con la domanda viene inviato al modello generativo:

```
llm = OllamaLLM(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
answer = llm.invoke(prompt.format(context=context, question=question))
```

4. Post-Processing e Risposta JSON
Il server formatta i metadati dei PDF e la risposta generata in un oggetto JSON pulito, includendo la confidenza matematica dell'operazione.

```
JSON
{
  "risposta_diretta": "Gli algoritmi di machine learning più usati sono...",
  "fonti": [
    {
      "file": "AI_as_a_Service.pdf",
      "pag": 1,
      "score": 0.696,
      "estratto": "MANNING Peter Elger Eóin Shanaghy Serverless machine..."
    }
  ],
  "confidenza": {
    "livello": "Media (Corrispondenza parziale)",
    "motivazione": "Punteggio medio 0.67."
  }
}
```