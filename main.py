import os
import glob
import logging

from contextlib import asynccontextmanager

# Importa le librerie per l'API web
from fastapi import FastAPI, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

# Importa le librerie per la logica RAG e AI (LangChain)
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
import qdrant_client

# --- CONFIGURAZIONE LOGGER ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE ---

# Caricamento file .env e uso di os.getenv()
# Invece di scrivere i parametri fissi nel codice, li legge dall'ambiente.
load_dotenv() 

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant:6333")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "default_collection")
SOURCE_DIR = os.getenv("SOURCE_DIR", "/app/documenti_da_indicizzare")

# Definisce le strategie di chunking per sperimentare diverse granularità
CHUNK_PRESETS = {
    "SMALL": {"size": 400, "overlap": 80},     # Chunk piccoli per dettagli precisi
    "MEDIUM": {"size": 1000, "overlap": 200},  # Bilanciamento standard
    "LARGE": {"size": 2000, "overlap": 300}    # Chunk grandi per contesti ampi
}

# Dizionario in memoria per tenere traccia dello stato delle collezioni attive
active_collections = {}

# --- FUNZIONI DI UTILITÀ ---

def get_confidence_level(score: float) -> str:
    """
    Traduce il punteggio numerico di similarità (da 0 a 1) in un'etichetta testuale leggibile.
    Serve per dare un feedback immediato sulla qualità del recupero dati.
    """
    if score > 0.75: return "Alta (Forte corrispondenza)"
    if score > 0.45: return "Media (Corrispondenza parziale)"
    return "Bassa (Poca pertinenza)"

def log_to_terminal(question: str, answer: str, confidenza: str, sources: list):
    """
    Stampa nel terminale del server un log dettagliato dell'interazione.
    Utile per il debug e per mostrare il funzionamento "dietro le quinte" durante la demo.
    """
    preview = answer[:300].replace('\n', ' ') + "..." if len(answer) > 300 else answer
    
    logger.info(f"RAG DOMANDA: {question}")
    logger.info(f"RAG RISPOSTA AI: {preview}")
    logger.info(f"RAG CONFIDENZA: {confidenza}")
    
    for src in sources:
        logger.info(f"RAG FONTE UTILIZZATA - [Score: {src['score']}] {src['file']} (Pag. {src['pag']})")

def ingest_local_documents(strategy: str = "MEDIUM"):
    """
    Gestisce l'intero processo di indicizzazione dei documenti (ETL Pipeline):
    1. Controlla se esiste già una cache su Qdrant.
    2. Se non esiste, carica i PDF dalla cartella.
    3. Esegue lo splitting (suddivisione) del testo in chunk.
    4. Calcola gli embeddings e salva tutto nel database vettoriale.
    """
    
    # 1. Connessione al Client Qdrant per verificare la presenza di dati
    client = qdrant_client.QdrantClient(url=QDRANT_HOST)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=EMBEDDING_MODEL)
    
    try:
        # Recupera la lista delle collezioni esistenti nel DB
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if exists:
            # Se la collezione esiste, evita di rifare il lavoro (Logica di Caching)
            count_info = client.count(collection_name=COLLECTION_NAME)
            logger.info(f"CACHE TROVATA: Collezione '{COLLECTION_NAME}' con {count_info.count} chunk nel DB.")
            logger.info("Salto l'ingestion per risparmiare tempo e risorse.")
            
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings,
            )
            # Registra la collezione come pronta all'uso
            active_collections[COLLECTION_NAME] = {
                "vector_store": vector_store,
                "strategy": strategy + " (Cached)"
            }
            return
    except Exception as e:
        logger.warning(f"Errore controllo cache: {e}. Procedo con ingestion completa.")

    # 2. Ingestion Reale (Se la cache non esiste)
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        return

    # Cerca tutti i file PDF nella directory specificata
    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        logger.warning("Nessun PDF trovato nella cartella locale.")
        return

    # Recupera i parametri di configurazione per il chunking
    config = CHUNK_PRESETS[strategy]
    all_chunks = []
    
    logger.info(f"INIZIO INGESTION (Strategia: {strategy}) - Parametri: Chunk Size {config['size']} | Overlap {config['overlap']}")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            filename = os.path.basename(pdf_path)
            
            # Caricamento del PDF (Parsing)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            
            # Suddivisione del testo (Chunking) usando separatori intelligenti
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["size"],
                chunk_overlap=config["overlap"],
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = text_splitter.split_documents(pages)
            
            # Calcola statistiche per monitorare la qualità del chunking
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
            
            logger.info(f"Elaborato file [{i}/{len(pdf_files)}] {filename} | Pagine: {len(pages)} | Chunk Generati: {len(chunks)} | Dim. Media: {int(avg_chunk_size)} caratteri")

            all_chunks.extend(chunks)
            
        except Exception as e:
            logger.error(f"Errore su {pdf_path}: {e}")

    # 3. Creazione Embeddings e Salvataggio (Vector Store)
    logger.info("Creazione Embeddings e Salvataggio in Qdrant in corso...")
    if all_chunks:
        # Converte il testo in vettori e li carica su Qdrant
        vector_store = QdrantVectorStore.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            url=QDRANT_HOST,
            collection_name=COLLECTION_NAME,
            force_recreate=True # Sovrascrive se necessario per aggiornare i dati
        )
        active_collections[COLLECTION_NAME] = {
            "vector_store": vector_store,
            "strategy": strategy
        }
        logger.info(f"SISTEMA PRONTO! Totale {len(all_chunks)} chunk indicizzati.")


# --- GESTIONE AVVIO E APP FASTAPI ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inizializzazione del sistema RAG...")
    ingest_local_documents() # Esegue il controllo/caricamento iniziale
    yield # Qui il server accetta le richieste
    logger.info("Chiusura del server in corso. Pulizia risorse...")

# Passiamo il lifespan durante la creazione dell'app
app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- ENDPOINT API ---

@app.post("/reindex")
async def trigger_reindex(background_tasks: BackgroundTasks):
    """
    Permette di caricare nuovi PDF a server già avviato. 
    Usa un BackgroundTask per non bloccare le altre chiamate web durante l'elaborazione.
    """
    logger.info("Avvio re-indicizzazione in background...")
    background_tasks.add_task(ingest_local_documents)
    return {"status": "ok", "dettaglio": "Re-indicizzazione avviata in background."}

@app.post("/ask")
async def ask(question: str = Form(...)):
    """
    Endpoint principale per le domande (RAG Pipeline).
    Riceve una domanda, cerca i documenti rilevanti e genera una risposta.
    """
    collection = "default_collection"
    
    # Verifica preliminare se il sistema è pronto
    if collection not in active_collections:
        raise HTTPException(status_code=404, detail="Il sistema sta ancora caricando i documenti.")
    
    info = active_collections[collection]
    vs = info["vector_store"]
    
    # 1. RETRIEVAL: Cerca i 4 chunk più simili alla domanda nel DB vettoriale
    results_with_score = vs.similarity_search_with_score(question, k=4)
    
    if not results_with_score:
        return {"risposta_diretta": "Nessuna informazione trovata.", "confidenza": "Nulla"}

    # Costruisce il contesto unendo il testo dei chunk trovati
    context = "\n".join([doc.page_content for doc, _ in results_with_score])
    
    # 2. PROMPTING: Definisce le istruzioni rigide per il modello di linguaggio
    prompt = PromptTemplate.from_template("""
    Rispondi in Italiano basandoti SOLO sul contesto fornito. 
    Se le informazioni non sono sufficienti, dillo chiaramente.
    Contesto: {context}
    Domanda: {question}
    Risposta:""")
    
    # 3. GENERATION: Invoca il modello LLM (Ollama) per generare la risposta
    llm = OllamaLLM(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
    answer = llm.invoke(prompt.format(context=context, question=question))
    
    # 4. POST-PROCESSING: Estrae i metadati delle fonti per la citazione
    sources = []
    total_score = 0
    for doc, score in results_with_score:
        total_score += score
        full_path = doc.metadata.get("source", "N/A")
        clean_filename = os.path.basename(full_path) 
        sources.append({
            "file": clean_filename,
            "pag": doc.metadata.get("page", 0) + 1,
            "score": round(score, 3),
            "estratto": doc.page_content[:150].replace("\n", " ") + "..."
        })
    
    # Calcola la confidenza media basata sulla similarità vettoriale
    avg_score = total_score / len(results_with_score)
    conf_level = get_confidence_level(avg_score)

    # Logga l'interazione per monitoraggio
    log_to_terminal(question, answer, conf_level, sources)
    
    # Restituisce il JSON strutturato come richiesto dalle specifiche
    return {
        "risposta_diretta": answer,
        "fonti": sources,
        "confidenza": {
            "livello": conf_level,
            "motivazione": f"Punteggio medio {round(avg_score, 2)}."
        }
    }