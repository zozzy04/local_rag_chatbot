"""Test degli endpoint API REST (§8.1).

Usa FastAPI TestClient. I test su /retrieve e /ask fanno mock
del vector store e del LLM per non richiedere Qdrant/Ollama attivi.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("GROUP_ID", "testGroup")
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("QDRANT_HOST", "http://localhost:6333")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app, raise_server_exceptions=False)

VALID_HEADERS = {
    "X-API-Key": "test-key",
    "X-API-Version": "1.0",
    "X-Group-Id": "testCaller",
}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "group_id" in data
        assert "timestamp" in data

    def test_health_no_auth_required(self):
        # /health non richiede headers
        resp = client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_info_requires_auth(self):
        resp = client.get("/info")
        assert resp.status_code == 422  # mancano gli header required

    def test_info_wrong_version(self):
        resp = client.get("/info", headers={
            "X-API-Key": "test-key",
            "X-API-Version": "0.9",
            "X-Group-Id": "x",
        })
        assert resp.status_code == 426
        assert resp.json()["error_code"] == "VERSION_MISMATCH"

    def test_info_wrong_api_key(self):
        resp = client.get("/info", headers={
            "X-API-Key": "wrong",
            "X-API-Version": "1.0",
            "X-Group-Id": "x",
        })
        assert resp.status_code == 401
        assert resp.json()["error_code"] == "UNAUTHORIZED"

    def test_info_valid(self):
        import sys
        from unittest.mock import MagicMock

        fake_qc_module = MagicMock()
        fake_qc_module.QdrantClient.return_value.get_collections.return_value.collections = []

        with patch.dict(sys.modules, {"qdrant_client": fake_qc_module}), \
             patch("api.server._get_rag_functions"):
            resp = client.get("/info", headers=VALID_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "group_id" in data
        assert "embedding_model" in data
        assert data["api_version"] == "1.0"


# ---------------------------------------------------------------------------
# /retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def _mock_vs_result(self):
        doc = MagicMock()
        doc.page_content = "Testo di esempio del chunk."
        doc.metadata = {"source": "test.pdf", "page": 1}
        return [(doc, 0.87)]

    def test_retrieve_valid(self):
        import sys

        fake_qc_module = MagicMock()
        col_mock = MagicMock()
        col_mock.name = "col"
        fake_qc_module.QdrantClient.return_value.get_collections.return_value.collections = [col_mock]
        fake_qc_rest = MagicMock()
        fake_qc_module.http.models = fake_qc_rest

        vs_instance = MagicMock()
        vs_instance.similarity_search_with_score.return_value = self._mock_vs_result()

        fake_lq = MagicMock()
        fake_lq.QdrantVectorStore.return_value = vs_instance

        fake_lo = MagicMock()

        with patch.dict(sys.modules, {
                "qdrant_client": fake_qc_module,
                "qdrant_client.http": fake_qc_module.http,
                "qdrant_client.http.models": fake_qc_rest,
                "langchain_qdrant": fake_lq,
                "langchain_ollama": fake_lo,
             }), \
             patch("api.server._get_rag_functions", return_value=(None, None, {}, "col")):
            resp = client.post(
                "/retrieve",
                json={"question": "Cos'è l'IA?", "k": 3},
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "chunks" in data
        assert "group_id" in data
        assert "latency_ms" in data

    def test_retrieve_invalid_k(self):
        resp = client.post(
            "/retrieve",
            json={"question": "test", "k": 0},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 422

    def test_retrieve_missing_question(self):
        resp = client.post(
            "/retrieve",
            json={"k": 3},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------

class TestAsk:
    def _mock_rag_result(self):
        return {
            "risposta_diretta": "L'IA è la simulazione dell'intelligenza umana.",
            "fonti": [{"file": "test.pdf", "pag": 1, "score": 0.87, "estratto": "...testo..."}],
            "confidenza": {"livello": "Alta (Forte corrispondenza)", "motivazione": "Score: 0.87"},
        }

    def test_ask_valid(self):
        with patch("api.server._get_rag_functions") as mock_fn:
            mock_core = MagicMock(return_value=self._mock_rag_result())
            mock_fn.return_value = (mock_core, None, {}, "col")
            resp = client.post(
                "/ask",
                json={"question": "Cos'è l'IA?", "k": 4},
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"]["confidence"] in ("alta", "media", "bassa")
        assert "is_abstention" in data["answer"]
        assert data["group_id"] == "testGroup"

    def test_ask_abstention(self):
        abstention_result = {
            "risposta_diretta": "ASTENSIONE: Non ho trovato la risposta.",
            "fonti": [],
            "confidenza": {"livello": "Bassa (Astensione forzata)", "motivazione": ""},
        }
        with patch("api.server._get_rag_functions") as mock_fn:
            mock_core = MagicMock(return_value=abstention_result)
            mock_fn.return_value = (mock_core, None, {}, "col")
            resp = client.post(
                "/ask",
                json={"question": "Domanda senza risposta"},
                headers=VALID_HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]["is_abstention"] is True
        assert data["answer"]["confidence"] == "bassa"

    def test_ask_missing_question(self):
        resp = client.post("/ask", json={}, headers=VALID_HEADERS)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Test di integrazione federata (stub — §7.5)
# ---------------------------------------------------------------------------

class TestFederatedIntegration:
    """
    Test di integrazione con un peer simulato via httpx mock.
    Verifica che il sistema gestisca correttamente peer offline (§FAQ resilience).
    """

    def test_offline_peer_graceful_degradation(self):
        """Un peer non raggiungibile non deve far fallire la query federata."""
        import asyncio
        from api.federation import federated_retrieve_async

        peers = [{"id": "groupZ", "base_url": "http://localhost:19999", "enabled": True}]
        local_chunks = [
            {"chunk_id": "c1", "text": "testo locale", "source": "local.pdf", "page": 1, "score": 0.8}
        ]

        merged, errors = asyncio.run(
            federated_retrieve_async("domanda test", 5, local_chunks, peers=peers)
        )
        assert "groupZ" in errors
        assert len(merged) >= 1  # almeno i chunk locali

    def test_rrf_consistent_ordering(self):
        """Verifica che RRF ordini correttamente chunk duplicati tra gruppi."""
        from api.federation import reciprocal_rank_fusion

        group_results = {
            "local": [{"chunk_id": "c_shared", "text": "t", "source": "x.pdf", "page": 1, "score": 0.9}],
            "remote": [
                {"chunk_id": "c_other", "text": "t2", "source": "y.pdf", "page": 2, "score": 0.95},
                {"chunk_id": "c_shared", "text": "t", "source": "x.pdf", "page": 1, "score": 0.85},
            ],
        }
        merged = reciprocal_rank_fusion(group_results)
        ids = [c["chunk_id"] for c in merged]
        # c_shared appare in entrambi → deve avere rank più alto di c_other (solo in remote)
        assert ids.index("c_shared") < ids.index("c_other")
