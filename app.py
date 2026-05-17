import argparse
import sys
import json
import csv
import os


def _load_rag():
    """Import lazy delle funzioni RAG (richiedono Ollama/Qdrant installati)."""
    try:
        from app.main import ingest_local_documents, core_rag_ask
        return ingest_local_documents, core_rag_ask
    except ImportError as e:
        print(f"Errore importazione moduli RAG: {e}")
        print("Assicurati che le dipendenze siano installate (requirements.txt).")
        sys.exit(1)


def print_rag_response(result_dict: dict):
    """Formatta l'output CLI come richiesto al punto 3.5."""
    print("\n" + "=" * 60)
    print("[RISPOSTA]")
    print(result_dict.get("risposta_diretta", ""))

    print("\n[FONTI]")
    for f in result_dict.get("fonti", []):
        group_tag = f" [gruppo: {f['group_id']}]" if f.get("group_id") else ""
        print(f"- file:{group_tag} {f['file']}, pag: {f['pag']}, score: {f['score']}")
        print(f"  estratto: \"{f['estratto']}\"")

    conf = result_dict.get("confidenza", {})
    print(f"\n[CONFIDENZA] {conf.get('livello', 'N/A')}")
    print(f"[MOTIVAZIONE] {conf.get('motivazione', 'N/A')}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# DB commands
# ---------------------------------------------------------------------------

def cmd_db_init():
    """Inizializza lo schema DB tramite Alembic e inserisce il gruppo locale."""
    print("Esecuzione migrazioni Alembic...")
    try:
        from db.database import run_migrations
        run_migrations()
        print("Migrazioni completate.")
    except Exception as e:
        print(f"ERRORE durante le migrazioni: {e}")
        sys.exit(1)

    # Crea il record del gruppo locale se non esiste
    try:
        from db.database import get_db_session
        from db.models import Group
        group_id = os.getenv("GROUP_ID", "groupA")
        group_name = os.getenv("GROUP_NAME", "Team RAG")

        with get_db_session() as session:
            existing = session.get(Group, group_id)
            if not existing:
                from datetime import datetime
                session.add(Group(
                    id=group_id,
                    name=group_name,
                    description=os.getenv("GROUP_DESCRIPTION", "Corpus su AI, Agents e Cloud"),
                    created_at=datetime.utcnow()
                ))
                print(f"Gruppo '{group_id}' creato nel DB.")

        # Seed chunking configs
        _seed_chunking_configs()

        print(f"DB inizializzato con successo. URL: {os.getenv('DATABASE_URL', 'sqlite:///./rag.db')}")
    except Exception as e:
        print(f"ERRORE durante l'init del gruppo: {e}")
        sys.exit(1)


def _seed_chunking_configs():
    from db.database import get_db_session
    from db.models import ChunkingConfig
    presets = [
        ("SMALL", 400, 80, "[\\n\\n, \\n, ., ]"),
        ("MEDIUM", 1000, 200, "[\\n\\n, \\n, ., ]"),
        ("LARGE", 2000, 300, "[\\n\\n, \\n, ., ]"),
    ]
    with get_db_session() as session:
        for name, size, overlap, seps in presets:
            if not session.query(ChunkingConfig).filter_by(name=name).first():
                session.add(ChunkingConfig(name=name, chunk_size=size, overlap=overlap, separators=seps))
    print("Configurazioni di chunking (SMALL/MEDIUM/LARGE) registrate nel DB.")


def cmd_db_history(last: int = 20):
    """Mostra le ultime N query e risposte salvate nel DB."""
    try:
        from db.database import SessionLocal
        from db.models import Query, Answer
        session = SessionLocal()

        rows = (
            session.query(Query, Answer)
            .join(Answer, Answer.query_id == Query.id, isouter=True)
            .order_by(Query.asked_at.desc())
            .limit(last)
            .all()
        )
        session.close()

        if not rows:
            print("Nessuna query registrata nel DB ancora. Fai qualche domanda con 'python app.py ask'.")
            return

        print(f"\n{'=' * 80}")
        print(f"ULTIME {last} QUERY — Gruppo: {os.getenv('GROUP_ID', 'groupA')}")
        print(f"{'=' * 80}")
        for q, a in rows:
            print(f"[{q.asked_at}] (ID={q.id}, k={q.k}, federated={q.federated})")
            print(f"  Q: {q.question[:120]}")
            if a:
                conf = a.confidence or "N/A"
                ans_preview = (a.answer_text or "")[:200].replace("\n", " ")
                print(f"  A: [{conf.upper()}] {ans_preview}...")
            print(f"  Latency: {q.latency_ms or 0} ms")
            print()

    except Exception as e:
        print(f"ERRORE: {e}. Hai eseguito 'python app.py db init'?")


def cmd_db_export(format: str = "csv"):
    """Esporta la storia query/risposte in CSV o JSON."""
    try:
        from db.database import SessionLocal
        from db.models import Query, Answer, AnswerSource

        session = SessionLocal()
        rows = (
            session.query(Query, Answer)
            .join(Answer, Answer.query_id == Query.id, isouter=True)
            .order_by(Query.asked_at.desc())
            .all()
        )
        session.close()

        data = [
            {
                "query_id": q.id,
                "group_id": q.group_id,
                "question": q.question,
                "k": q.k,
                "source_filter": q.source_filter,
                "federated": q.federated,
                "asked_at": str(q.asked_at),
                "latency_ms": q.latency_ms,
                "answer_text": a.answer_text if a else "",
                "confidence": a.confidence if a else "",
                "model_name": a.model_name if a else "",
            }
            for q, a in rows
        ]

        outfile = f"history_export.{format}"
        if format == "json":
            with open(outfile, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            if data:
                with open(outfile, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)

        print(f"Esportati {len(data)} record in '{outfile}'.")

    except Exception as e:
        print(f"ERRORE: {e}")


# ---------------------------------------------------------------------------
# Analytics queries (§6.2)
# ---------------------------------------------------------------------------

def cmd_db_analyze():
    """Esegue le 3 query analitiche richieste dal §6.2."""
    try:
        from db.database import SessionLocal
        session = SessionLocal()
        print("\n" + "=" * 60)
        print("QUERY ANALITICA 1 — Top 10 domande più frequenti")
        print("=" * 60)
        from db.analytics import top10_frequent_questions
        for row in top10_frequent_questions(session):
            print(f"  [{row.group_id}] (n={row.count}) {row.question[:80]}")

        print("\n" + "=" * 60)
        print("QUERY ANALITICA 3 — Latenza media per configurazione chunking")
        print("=" * 60)
        from db.analytics import avg_latency_by_chunking
        for row in avg_latency_by_chunking(session):
            print(f"  Preset: {row[0]:10s} | Avg: {row[1]} ms | N queries: {row[2]}")

        session.close()
    except Exception as e:
        print(f"ERRORE analytics: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sistema RAG CLI — Progetto GenAI")
    subparsers = parser.add_subparsers(dest="command", help="Comandi disponibili")

    # --- index ---
    parser_index = subparsers.add_parser("index", help="Indicizza i PDF del proprio gruppo")
    parser_index.add_argument(
        "--preset", type=str, default="MEDIUM", choices=["SMALL", "MEDIUM", "LARGE"],
        help="Preset di chunking (default: MEDIUM)"
    )

    # --- ask ---
    parser_ask = subparsers.add_parser("ask", help="Interroga la Knowledge Base")
    parser_ask.add_argument("domanda", type=str, help="La domanda in linguaggio naturale")
    parser_ask.add_argument("--source", type=str, default=None, help="Filtra su un singolo file PDF")
    parser_ask.add_argument("-k", type=int, default=4, help="Top-K chunk (default: 4)")
    parser_ask.add_argument(
        "--federated", action="store_true",
        help="Interroga anche le KB remote registrate in peers.yaml"
    )
    parser_ask.add_argument(
        "--remote", type=str, default=None,
        help="Interroga KB remote specifiche (es. groupB,groupC)"
    )
    # §5 Extension flags
    parser_ask.add_argument("--multi-query", action="store_true", help="[Ext §5 #3] Multi-query expansion")
    parser_ask.add_argument("--hybrid", action="store_true", help="[Ext §5 #2] Hybrid BM25+dense search")
    parser_ask.add_argument("--cache", action="store_true", help="[Ext §5 #8] Semantic query cache")
    parser_ask.add_argument("--self-correct", action="store_true", help="[Ext §5 #9] Self-correction loop")

    # --- serve ---
    parser_serve = subparsers.add_parser("serve", help="Espone la KB come API REST (§7.2)")
    parser_serve.add_argument("--port", type=int, default=8000, help="Porta del server (default: 8000)")
    parser_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")

    # --- db ---
    parser_db = subparsers.add_parser("db", help="Gestione del database relazionale")
    db_sub = parser_db.add_subparsers(dest="db_command", help="Sottocomandi DB")

    db_sub.add_parser("init", help="Inizializza lo schema DB (Alembic migrations)")

    parser_hist = db_sub.add_parser("history", help="Mostra le ultime query")
    parser_hist.add_argument("--last", type=int, default=20, help="Numero di record da mostrare")

    parser_export = db_sub.add_parser("export", help="Esporta la storia query in CSV o JSON")
    parser_export.add_argument("--format", type=str, default="csv", choices=["csv", "json"])

    db_sub.add_parser("analyze", help="Esegue le 3 query analitiche richieste dal §6.2")

    args = parser.parse_args()

    # --- Routing ---
    if args.command == "index":
        ingest_local_documents, _ = _load_rag()
        print(f"Avvio indicizzazione con preset: {args.preset}")
        ingest_local_documents(strategy=args.preset)

    elif args.command == "ask":
        _, core_rag_ask = _load_rag()
        federated = args.federated
        remote_peers = None

        if args.remote:
            from api.federation import load_peers
            all_peers = load_peers()
            remote_ids = [r.strip() for r in args.remote.split(",")]
            remote_peers = [p for p in all_peers if p["id"] in remote_ids]
            if not remote_peers:
                print(f"Attenzione: nessun peer trovato tra {remote_ids} in peers.yaml")

        ext_flags = []
        if getattr(args, 'multi_query', False): ext_flags.append("multi-query")
        if getattr(args, 'hybrid', False): ext_flags.append("hybrid-search")
        if getattr(args, 'cache', False): ext_flags.append("cache")
        if getattr(args, 'self_correct', False): ext_flags.append("self-correction")

        print(f"Interrogazione RAG (K={args.k}, fonte={args.source or 'tutte'}, federata={federated or bool(args.remote)}, ext={ext_flags or 'none'})")
        try:
            result = core_rag_ask(
                question=args.domanda,
                top_k=args.k,
                source_filter=args.source,
                federated=federated,
                remote_peers=remote_peers,
                use_multi_query=getattr(args, 'multi_query', False),
                use_hybrid_search=getattr(args, 'hybrid', False),
                use_cache=getattr(args, 'cache', False),
                use_self_correction=getattr(args, 'self_correct', False),
            )
            print_rag_response(result)
        except ValueError as e:
            print(f"\nERRORE: {e}")
        except Exception as e:
            print(f"\nERRORE INASPETTATO: {e}")

    elif args.command == "serve":
        import uvicorn
        print(f"Avvio API server su http://{args.host}:{args.port}")
        print("Endpoints disponibili: /health  /info  /documents  /retrieve  /ask  /docs")
        uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)

    elif args.command == "db":
        if args.db_command == "init":
            cmd_db_init()
        elif args.db_command == "history":
            cmd_db_history(last=args.last)
        elif args.db_command == "export":
            cmd_db_export(format=args.format)
        elif args.db_command == "analyze":
            cmd_db_analyze()
        else:
            parser_db.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
