import argparse
import sys
import json

# Importa dal main.py le funzioni core
try:
    from app.main import ingest_local_documents, core_rag_ask
except ImportError:
    print("Errore: impossibile importare da main.py. Assicurati di essere nella cartella giusta.")
    sys.exit(1)

def print_rag_response(result_dict: dict):
    """Formatta l'output per la CLI come richiesto dal professore al punto 3.5"""
    print("\n" + "="*50)
    print("[RISPOSTA]")
    print(result_dict.get("risposta_diretta", ""))
    
    print("\n[FONTI]")
    for f in result_dict.get("fonti", []):
        print(f"- file: {f['file']}, pag: {f['pag']}, score: {f['score']}")
        print(f"  estratto: \"{f['estratto']}\"")
        
    conf = result_dict.get("confidenza", {})
    print(f"\n[CONFIDENZA] {conf.get('livello', 'N/A')}")
    print(f"[MOTIVAZIONE] {conf.get('motivazione', 'N/A')}")
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Sistema RAG CLI - Progetto GenAI")
    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # --- COMANDO: index ---
    parser_index = subparsers.add_parser("index", help="Indicizza i PDF del proprio gruppo")
    parser_index.add_argument("--preset", type=str, default="MEDIUM", choices=["SMALL", "MEDIUM", "LARGE"], help="Preset di chunking da utilizzare")

    # --- COMANDO: ask ---
    parser_ask = subparsers.add_parser("ask", help="Interroga la Knowledge Base")
    parser_ask.add_argument("domanda", type=str, help="La domanda in linguaggio naturale")
    parser_ask.add_argument("--source", type=str, default=None, help="Filtra la ricerca su un singolo file PDF")
    parser_ask.add_argument("-k", type=int, default=4, help="Numero di chunk da recuperare (Top-K)")

    args = parser.parse_args()

    if args.command == "index":
        print(f"Avvio indicizzazione con preset: {args.preset}")
        ingest_local_documents(strategy=args.preset)

    elif args.command == "ask":
        print(f"Interrogazione RAG in corso... (Top-K: {args.k}, Fonte: {args.source if args.source else 'Tutte'})")
        try:
            # Chiama la logica condivisa
            result = core_rag_ask(question=args.domanda, top_k=args.k, source_filter=args.source)
            # Stampa il risultato formattato
            print_rag_response(result)
        except ValueError as e:
            print(f"\nERRORE: {e}")
        except Exception as e:
            print(f"\nERRORE INASPETTATO: {e}")

    # ... (resto dei routing) ...

if __name__ == "__main__":
    main()