"""3 query analitiche richieste dal §6.2 del README."""
from sqlalchemy import text, func, desc
from sqlalchemy.orm import Session

from db.models import Query, Answer, Document, AnswerSource


def top10_frequent_questions(session: Session, group_id: str | None = None):
    """Top 10 domande più frequenti per gruppo."""
    q = session.query(
        Query.question,
        Query.group_id,
        func.count(Query.id).label("count")
    )
    if group_id:
        q = q.filter(Query.group_id == group_id)
    return (
        q.group_by(Query.question, Query.group_id)
        .order_by(desc("count"))
        .limit(10)
        .all()
    )


def confidence_distribution(session: Session):
    """Distribuzione della confidenza per file sorgente."""
    return (
        session.query(
            Document.filename,
            Answer.confidence,
            func.count(Answer.id).label("count")
        )
        .join(AnswerSource, AnswerSource.document_id == Document.id)
        .join(Answer, Answer.id == AnswerSource.answer_id)
        .group_by(Document.filename, Answer.confidence)
        .order_by(Document.filename, Answer.confidence)
        .all()
    )


def avg_latency_by_chunking(session: Session):
    """Tempo medio di risposta (ms) per configurazione di chunking."""
    return session.execute(
        text("""
            SELECT cc.name            AS chunking_config,
                   ROUND(AVG(q.latency_ms), 1) AS avg_latency_ms,
                   COUNT(q.id)        AS n_queries
            FROM   queries q
            JOIN   documents d  ON d.group_id = q.group_id
            JOIN   chunking_configs cc ON cc.id = d.chunking_config_id
            GROUP  BY cc.name
            ORDER  BY avg_latency_ms ASC
        """)
    ).fetchall()
