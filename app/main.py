import os
import glob
import logging
import hashlib
from pathlib import Path
from contextlib import asynccontextmanager

import yaml
from pathlib import Path

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
from qdrant_client.http import models as rest

from app.prompt import rag_prompt, rewrite_prompt

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
SOURCE_DIR = os.getenv("SOURCE_DIR", "/data/input")

# 1. Trova il percorso ASSOLUTO in cui si trova questo file (main.py)
BASE_DIR = Path(__file__).resolve().parent

# 2. Uniscilo al nome del file
CONFIG_PATH = BASE_DIR / "config.yaml"

try:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configurazione caricata con successo da: {CONFIG_PATH}") # Aggiungiamo questa riga per conferma!
except FileNotFoundError:
    logger.error(f"ERRORE CRITICO: File config.yaml NON trovato nel percorso esatto: {CONFIG_PATH}")
    config = {}

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

def calculate_file_hash(filepath: str) -> str:
    """
    Calcola l'hash SHA256 di un file.
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Legge il file in blocchi per non sovraccaricare la RAM con file grandi
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_chunking_config(strategy: str) -> dict:
    """Carica la configurazione di chunking dal file YAML."""
    # Usiamo la variabile globale CONFIG_PATH definita in alto nel file!
    if not CONFIG_PATH.exists():
        logger.warning(f"File {CONFIG_PATH} non trovato. Uso configurazione MEDIUM di default.")
        return {"chunk_size": 1000, "chunk_overlap": 200}
    
    with open(CONFIG_PATH, 'r') as file:
        config_data = yaml.safe_load(file)
        
    presets = config_data.get("chunking_presets", {})
    if strategy not in presets:
        logger.warning(f"Preset '{strategy}' non trovato in {CONFIG_PATH}. Uso MEDIUM.")
        return presets.get("MEDIUM", {"chunk_size": 1000, "chunk_overlap": 200})
        
    return presets[strategy]

def is_file_in_qdrant(client: qdrant_client.QdrantClient, collection_name: str, file_hash: str) -> bool:
    """
    Verifica se esistono vettori associati a questo hash nella collezione Qdrant.
    """
    try:
        # Recupera le collezioni per verificare che non stiamo interrogando una collezione inesistente
        collections = client.get_collections().collections
        if not any(c.name == collection_name for c in collections):
            return False

        # Fa una ricerca filtrando solo per il metadato 'file_hash'
        records, next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="metadata.file_hash",
                        match=rest.MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1, # Ci basta sapere se c'è almeno 1 record
            with_payload=False, # Non ci serve il payload (testo), solo sapere se esiste
            with_vectors=False
        )
        return len(records) > 0
    except Exception as e:
        logger.warning(f"Errore durante il controllo hash in Qdrant: {e}")
        return False

def ingest_local_documents(strategy: str = "MEDIUM", embedding_model_name: str = None):
    """
    Gestisce l'intero processo di indicizzazione dei documenti.
    """
    emb_model = embedding_model_name or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # --- ISOLAMENTO DELLA COLLEZIONE per experiments.py ---
    # Se è un esperimento, usiamo una collezione temporanea per non sporcare il DB principale
    target_collection = f"{COLLECTION_NAME}_experiment" if embedding_model_name else COLLECTION_NAME
    
    # 1. Connessione al Client Qdrant
    client = qdrant_client.QdrantClient(url=QDRANT_HOST)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=emb_model)
    
    if embedding_model_name:
        vector_size = 768 if "nomic" in emb_model.lower() else 1024
        logger.info(f"Modalità Esperimento: Ricreazione collezione Qdrant '{target_collection}' con size {vector_size}")
        client.recreate_collection(
            collection_name=target_collection,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        if target_collection in active_collections:
            del active_collections[target_collection]

    # Verifica se la collezione esiste già nel DB
    collections = client.get_collections().collections
    collection_exists = any(c.name == target_collection for c in collections)
    
    # Se esiste, inizializziamo subito il vector store
    if collection_exists:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=target_collection,
            embedding=embeddings,
        )
        active_collections[target_collection] = {
            "vector_store": vector_store,
            "strategy": strategy
        }
    else:
        logger.info(f"La collezione '{target_collection}' non esiste ancora. Verrà creata a breve.")
        vector_store = None

    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        return

    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        logger.warning("Nessun PDF trovato nella cartella locale.")
        return

    preset_config = load_chunking_config(strategy)
    config = {
        "size": preset_config.get("chunk_size", 1000),
        "overlap": preset_config.get("chunk_overlap", 200)
    }
    separators = preset_config.get("separators", ["\n\n", "\n", ".", " "])
    
    new_chunks_to_add = []
    logger.info(f"INIZIO INGESTION (Strategia: {strategy} | Modello: {emb_model} | Collezione: {target_collection})")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            filename = os.path.basename(pdf_path)
            file_hash = calculate_file_hash(pdf_path)
            
            # Controllo sull'hash della collezione corretta
            if collection_exists and is_file_in_qdrant(client, target_collection, file_hash):
                logger.info(f"File [{i}/{len(pdf_files)}]: {filename} GIÀ INDICIZZATO. Salto.")
                continue 
            
            logger.info(f"Elaborazione file [{i}/{len(pdf_files)}]: {filename} (Hash: {file_hash[:8]}...)")
            
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            for idx, page in enumerate(pages):
                 page.metadata = {
                     "source": filename,
                     "page": page.metadata.get('page', idx) + 1,
                     "file_hash": file_hash 
                 }
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["size"],
                chunk_overlap=config["overlap"],
                separators=separators
            )
            chunks = text_splitter.split_documents(pages)
            
            new_chunks_to_add.extend(chunks)
            
        except Exception as e:
            logger.error(f"Errore su {pdf_path}: {e}")

    # --- 3. CREAZIONE COLLEZIONE O AGGIUNTA CHUNK ---
    if new_chunks_to_add:
        if not collection_exists:
            logger.info(f"Creazione della nuova collezione '{target_collection}' e salvataggio di {len(new_chunks_to_add)} chunk...")
            
            vector_store = QdrantVectorStore.from_documents(
                documents=new_chunks_to_add,
                embedding=embeddings,
                url=QDRANT_HOST,
                collection_name=target_collection,
                force_recreate=True 
            )
            active_collections[target_collection] = {
                "vector_store": vector_store,
                "strategy": strategy
            }
        else:
            logger.info(f"Aggiunta di {len(new_chunks_to_add)} nuovi chunk alla collezione esistente '{target_collection}'...")
            vector_store.add_documents(new_chunks_to_add)

        logger.info("SISTEMA PRONTO! Indicizzazione completata.")
    else:
        logger.info("SISTEMA PRONTO! Nessun nuovo documento.")

def core_rag_ask(question: str, top_k: int = 4, source_filter: str = None, embedding_model_name: str = None):
    """
    Logica di business principale per il RAG.
    """
    emb_model = embedding_model_name or os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    # --- ISOLAMENTO DELLA COLLEZIONE ESPERIMENTI ---
    target_collection = f"{COLLECTION_NAME}_experiment" if embedding_model_name else COLLECTION_NAME
    
    # INIZIALIZZAZIONE "LAZY"
    if target_collection not in active_collections:
        logger.info(f"Inizializzazione 'lazy' della connessione al vector store ({target_collection})...")
        client = qdrant_client.QdrantClient(url=QDRANT_HOST)
        
        collections = client.get_collections().collections
        if any(c.name == target_collection for c in collections):
            embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=emb_model)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=target_collection,
                embedding=embeddings,
            )
            active_collections[target_collection] = {
                "vector_store": vector_store,
                "strategy": "Loaded from DB"
            }
        else:
            raise ValueError(f"La collezione '{target_collection}' non esiste in Qdrant.")
    
    info = active_collections[target_collection]
    vs = info["vector_store"]
    llm = OllamaLLM(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
     
    qdrant_filter = None
    if source_filter:
        qdrant_filter = rest.Filter(
            must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=source_filter))]
        )
    
    max_attempts = 2
    current_question = question
    final_answer = ""
    results_with_score = []
    
    for attempt in range(max_attempts):
        logger.info(f"Tentativo {attempt + 1}/{max_attempts} con query: '{current_question}'")
        
        results_with_score = vs.similarity_search_with_score(
            query=current_question, 
            k=top_k, 
            filter=qdrant_filter
        )
        
        if not results_with_score:
            final_answer = "ASTENSIONE: Nessun documento recuperato dal database."
            break

        context = "\n---\n".join([
            f"[File: {doc.metadata.get('source', 'N/A')} | Pag: {doc.metadata.get('page', 0)}] {doc.page_content}" 
            for doc, _ in results_with_score
        ])
        
        final_answer = llm.invoke(rag_prompt.format(context=context, question=current_question)).strip()
        
        if "ASTENSIONE:" not in final_answer:
            break
            
        if attempt < max_attempts - 1:
            logger.warning("Il modello si è astenuto. Riformulazione query in corso...")
            current_question = llm.invoke(rewrite_prompt.format(question=current_question)).strip()
            current_question = current_question.replace('"', '').replace("'", "")
    
    sources = []
    total_score = 0
    if results_with_score:
        for doc, score in results_with_score:
            total_score += score
            sources.append({
                "file": doc.metadata.get("source", "N/A"),
                "pag": doc.metadata.get("page", 0),
                "score": round(score, 3),
                "estratto": doc.page_content[:150].replace("\n", " ") + "..."
            })
        avg_score = total_score / len(results_with_score)
        conf_level = get_confidence_level(avg_score)
    else:
        avg_score = 0.0
        conf_level = "Bassa"

    if "ASTENSIONE:" in final_answer:
        final_answer = "ASTENSIONE: Mi dispiace, ma le informazioni presenti nei documenti non sono sufficienti."
        conf_level = "Bassa (Astensione forzata)"

    log_to_terminal(question, final_answer, conf_level, sources)
    
    return {
        "risposta_diretta": final_answer,
        "fonti": sources,
        "confidenza": {
            "livello": conf_level,
            "motivazione": f"Punteggio medio vettoriale: {round(avg_score, 2)}."
        }
    }

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
async def ask_endpoint(question: str = Form(...), k: int = Form(4), source: str = Form(None)):
    """
    Endpoint Web. Incanala la richiesta verso il motore RAG.
    """
    try:
        result = core_rag_ask(question=question, top_k=k, source_filter=source)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Errore API /ask: {e}")
        raise HTTPException(status_code=500, detail="Errore interno del server.")