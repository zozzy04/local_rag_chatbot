import os
import glob
import logging
import hashlib
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
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning(f"File {config_path} non trovato. Uso configurazione MEDIUM di default.")
        return {"size": 1000, "overlap": 200}
    
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
        
    presets = config_data.get("chunking_presets", {})
    if strategy not in presets:
        logger.warning(f"Preset '{strategy}' non trovato in {config_path}. Uso MEDIUM.")
        return presets.get("MEDIUM", {"size": 1000, "overlap": 200})
        
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

def ingest_local_documents(strategy: str = "MEDIUM"):
    """
    Gestisce l'intero processo di indicizzazione dei documenti (ETL Pipeline):
    1. Inizializza la connessione a Qdrant.
    2. Carica i PDF dalla cartella.
    3. Controlla l'hash per evitare di riprocessare file già indicizzati.
    4. Esegue lo splitting e aggiunge l'hash ai metadati.
    5. Calcola gli embeddings e salva le novità nel database vettoriale.
    """
    
    # 1. Connessione al Client Qdrant
    client = qdrant_client.QdrantClient(url=QDRANT_HOST)
    embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=EMBEDDING_MODEL)
    
    # Verifica se la collezione esiste già nel DB
    collections = client.get_collections().collections
    collection_exists = any(c.name == COLLECTION_NAME for c in collections)
    
    # Se esiste, inizializziamo subito il vector store per tenerlo pronto per la ricerca
    if collection_exists:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
        )
        active_collections[COLLECTION_NAME] = {
            "vector_store": vector_store,
            "strategy": strategy
        }
    else:
        logger.info(f"La collezione '{COLLECTION_NAME}' non esiste ancora. Verrà creata a breve.")
        vector_store = None

    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        return

    # Cerca tutti i file PDF nella directory specificata
    pdf_files = glob.glob(os.path.join(SOURCE_DIR, "*.pdf"))
    if not pdf_files:
        logger.warning("Nessun PDF trovato nella cartella locale.")
        return

    # Recupera i parametri di configurazione dal file YAML
    preset_config = load_chunking_config(strategy)
    config = {
        "size": preset_config.get("chunk_size", 1000),
        "overlap": preset_config.get("chunk_overlap", 200)
    }
    # recuperiamo anche i separators dal file yaml se ci sono, altrimenti usiamo un default
    separators = preset_config.get("separators", ["\n\n", "\n", ".", " "])
    
    new_chunks_to_add = []
    
    logger.info(f"INIZIO INGESTION (Strategia: {strategy}) - Parametri: Chunk Size {config['size']} | Overlap {config['overlap']}")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            filename = os.path.basename(pdf_path)

            # Calcolo dell'hash del file
            file_hash = calculate_file_hash(pdf_path)
            
            # Controllo Hash: lo facciamo SOLO se la collezione esiste già
            if collection_exists and is_file_in_qdrant(client, COLLECTION_NAME, file_hash):
                logger.info(f"File [{i}/{len(pdf_files)}]: {filename} GIÀ INDICIZZATO (Hash in cache). Salto.")
                continue 
            
            logger.info(f"Elaborazione file [{i}/{len(pdf_files)}]: {filename} (Hash: {file_hash[:8]}...)")
            
            # Caricamento del PDF (Parsing)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            # Normalizzazione dei Metadati
            for idx, page in enumerate(pages):
                 page.metadata = {
                     "source": filename,
                     "page": page.metadata.get('page', idx) + 1,
                     "file_hash": file_hash 
                 }
            
            # Suddivisione del testo (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["size"],
                chunk_overlap=config["overlap"],
                separators=["\n\n", "\n", ".", " "]
            )
            chunks = text_splitter.split_documents(pages)
            
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
            logger.info(f"Elaborato file {filename} | Pagine: {len(pages)} | Chunk Generati: {len(chunks)} | Dim. Media: {int(avg_chunk_size)} caratteri")

            new_chunks_to_add.extend(chunks)
            
        except Exception as e:
            logger.error(f"Errore su {pdf_path}: {e}")

    # --- 3. CREAZIONE COLLEZIONE O AGGIUNTA CHUNK ---
    if new_chunks_to_add:
        if not collection_exists:
            # CASO A: Il DB è vuoto, la collezione non esiste. 
            # vector_store al momento è None. Usiamo from_documents per creare tutto da zero.
            logger.info(f"Creazione della nuova collezione '{COLLECTION_NAME}' e salvataggio di {len(new_chunks_to_add)} chunk...")
            
            vector_store = QdrantVectorStore.from_documents(
                documents=new_chunks_to_add,
                embedding=embeddings,
                url=QDRANT_HOST,
                collection_name=COLLECTION_NAME,
                force_recreate=True 
            )
            # Registra il vector_store appena creato
            active_collections[COLLECTION_NAME] = {
                "vector_store": vector_store,
                "strategy": strategy
            }
        else:
            # CASO B: La collezione esiste già (vector_store NON è None).
            # Facciamo solo l'aggiunta (append) dei nuovi file.
            logger.info(f"Aggiunta di {len(new_chunks_to_add)} nuovi chunk alla collezione esistente...")
            vector_store.add_documents(new_chunks_to_add)

        logger.info("SISTEMA PRONTO! Indicizzazione completata con successo.")
    else:
        logger.info("SISTEMA PRONTO! Nessun nuovo documento da indicizzare, i dati sono già allineati.")

def core_rag_ask(question: str, top_k: int = 4, source_filter: str = None) -> dict:
    """
    Logica di business principale per il RAG.
    Viene usata sia dalla CLI che dall'endpoint FastAPI.
    """
    collection_name = COLLECTION_NAME
    
    # INIZIALIZZAZIONE "LAZY"
    if collection_name not in active_collections:
        logger.info("Inizializzazione 'lazy' della connessione al vector store...")
        client = qdrant_client.QdrantClient(url=QDRANT_HOST)
        
        # Verifica se la collezione esiste effettivamente in Qdrant
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            embeddings = OllamaEmbeddings(base_url=OLLAMA_HOST, model=EMBEDDING_MODEL)
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )
            # Popola il dizionario locale
            active_collections[collection_name] = {
                "vector_store": vector_store,
                "strategy": "Loaded from DB"
            }
        else:
            raise ValueError(f"La collezione '{collection_name}' non esiste in Qdrant. Esegui prima l'indicizzazione ('python app.py index').")
    
    info = active_collections[collection_name]
    vs = info["vector_store"]
    llm = OllamaLLM(base_url=OLLAMA_HOST, model=OLLAMA_MODEL)
    
    # Costruzione del filtro opzionale
    qdrant_filter = None
    if source_filter:
        qdrant_filter = rest.Filter(
            must=[rest.FieldCondition(key="metadata.source", match=rest.MatchValue(value=source_filter))]
        )
    
    # --- LOOP DI RICERCA E RIFORMULAZIONE (Max 2 tentativi) ---
    max_attempts = 2
    current_question = question
    final_answer = ""
    results_with_score = []
    
    for attempt in range(max_attempts):
        logger.info(f"Tentativo {attempt + 1}/{max_attempts} con query: '{current_question}'")
        
        # 1. Retrieval
        results_with_score = vs.similarity_search_with_score(
            query=current_question, 
            k=top_k, 
            filter=qdrant_filter
        )
        
        # Se non trova assolutamente nulla (es. filtro sbagliato), si ferma
        if not results_with_score:
            final_answer = "ASTENSIONE: Nessun documento recuperato dal database."
            break

        # 2. Prepara il contesto e invoca l'LLM con il prompt rigido
        context = "\n---\n".join([
            f"[File: {doc.metadata.get('source', 'N/A')} | Pag: {doc.metadata.get('page', 0)}] {doc.page_content}" 
            for doc, _ in results_with_score
        ])
        
        final_answer = llm.invoke(rag_prompt.format(context=context, question=current_question)).strip()
        
        # 3. Controllo dell'Astensione
        if "ASTENSIONE:" not in final_answer:
            # L'LLM ha trovato la risposta, usciamo dal loop!
            break
            
        # l'LLM si è astenuto. Riformula la query se non è l'ultimo tentativo.
        if attempt < max_attempts - 1:
            logger.warning("Il modello si è astenuto. Riformulazione query in corso...")
            current_question = llm.invoke(rewrite_prompt.format(question=current_question)).strip()
            # Rimuove eventuali virgolette aggiunte dall'LLM
            current_question = current_question.replace('"', '').replace("'", "")
    
    # 4. POST-PROCESSING DELLE FONTI E FORMATTAZIONE
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

    # Se dopo i tentativi l'LLM si astiene ancora, aggiusta la risposta per l'utente
    if "ASTENSIONE:" in final_answer:
        final_answer = "Mi dispiace, ma le informazioni presenti nei documenti indicizzati non sono sufficienti per rispondere a questa domanda."
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