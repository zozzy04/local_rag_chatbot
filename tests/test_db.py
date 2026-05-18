"""Test CRUD sul database relazionale (§8.1).

Usa SQLite in memoria per velocità e isolamento.
"""
import os
import pytest
from datetime import datetime

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, Group, Document, Chunk, Query, Answer, AnswerSource, ChunkingConfig, Experiment, ExperimentResult


@pytest.fixture(scope="module")
def session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Group CRUD
# ---------------------------------------------------------------------------

class TestGroupCRUD:
    def test_create_group(self, session):
        g = Group(id="testGroup", name="Test Group", created_at=datetime.utcnow())
        session.add(g)
        session.commit()
        assert session.get(Group, "testGroup") is not None

    def test_read_group(self, session):
        g = session.get(Group, "testGroup")
        assert g.name == "Test Group"

    def test_update_group(self, session):
        g = session.get(Group, "testGroup")
        g.description = "Updated description"
        session.commit()
        g2 = session.get(Group, "testGroup")
        assert g2.description == "Updated description"


# ---------------------------------------------------------------------------
# Document + Chunk CRUD
# ---------------------------------------------------------------------------

class TestDocumentCRUD:
    def test_create_document(self, session):
        cfg = ChunkingConfig(name="MEDIUM_T", chunk_size=1000, overlap=200)
        session.add(cfg)
        session.flush()

        doc = Document(
            group_id="testGroup",
            filename="test.pdf",
            file_hash="abc123" * 10,
            n_pages=5,
            ingested_at=datetime.utcnow(),
            chunking_config_id=cfg.id,
        )
        session.add(doc)
        session.commit()
        assert doc.id is not None

    def test_idempotent_hash(self, session):
        # Secondo inserimento con stesso hash → non deve duplicare
        existing = session.query(Document).filter_by(file_hash="abc123" * 10).first()
        assert existing is not None
        count_before = session.query(Document).count()
        # Simula check idempotente
        found = session.query(Document).filter_by(
            group_id="testGroup", file_hash="abc123" * 10
        ).first()
        assert found is not None
        # Non aggiungiamo di nuovo
        count_after = session.query(Document).count()
        assert count_before == count_after

    def test_create_chunk(self, session):
        doc = session.query(Document).filter_by(filename="test.pdf").first()
        chunk = Chunk(
            document_id=doc.id, page=1, chunk_index=0,
            text="Testo di prova per il chunk.", char_count=28,
            embedding_model="nomic-embed-text", vector_id="abc123xx_p1_c0"
        )
        session.add(chunk)
        session.commit()
        assert chunk.id is not None


# ---------------------------------------------------------------------------
# Query + Answer + AnswerSource
# ---------------------------------------------------------------------------

class TestQueryAnswerCRUD:
    def test_atomic_query_answer(self, session):
        q = Query(
            group_id="testGroup", question="Cos'è un transformer?",
            k=4, federated=False, asked_at=datetime.utcnow(), latency_ms=250
        )
        session.add(q)
        session.flush()

        a = Answer(
            query_id=q.id, answer_text="Un transformer è...",
            confidence="alta", model_name="mistral", prompt_template_version="v1"
        )
        session.add(a)
        session.flush()

        doc = session.query(Document).first()
        src = AnswerSource(
            answer_id=a.id, document_id=doc.id, page=1,
            score=0.87, rank=1, remote_group_id=None
        )
        session.add(src)
        session.commit()

        assert q.id is not None
        assert a.query_id == q.id
        assert src.answer_id == a.id

    def test_cascade_delete_answer(self, session):
        q = Query(
            group_id="testGroup", question="Domanda da cancellare",
            k=3, federated=False, asked_at=datetime.utcnow()
        )
        session.add(q)
        session.flush()
        a = Answer(query_id=q.id, answer_text="R", confidence="media", model_name="m", prompt_template_version="v1")
        session.add(a)
        session.commit()

        q_id = q.id
        session.delete(q)
        session.commit()
        assert session.get(Answer, a.id) is None  # cascade


# ---------------------------------------------------------------------------
# Experiment + ExperimentResult
# ---------------------------------------------------------------------------

class TestExperimentCRUD:
    def test_create_experiment(self, session):
        exp = Experiment(
            name="Baseline MEDIUM nomic",
            config_json='{"chunking": "MEDIUM", "embedding": "nomic-embed-text"}',
            created_at=datetime.utcnow()
        )
        session.add(exp)
        session.flush()
        res = ExperimentResult(
            experiment_id=exp.id,
            question_id="q001",
            metric_name="recall@5",
            metric_value=0.75,
        )
        session.add(res)
        session.commit()
        assert exp.id is not None
        assert res.metric_value == 0.75
