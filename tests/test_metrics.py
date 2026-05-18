"""Test manuali delle funzioni di metrica (almeno una implementata a mano, §FAQ).

Copertura: Recall@k, MRR, Hit Rate, RRF, normalize_confidence.
"""
import pytest


# ---------------------------------------------------------------------------
# Funzioni di metrica implementate a mano (non via RAGAS wrapper)
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in relevant_ids if r in top_k)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: list) -> float:
    """MRR: 1/rank of the first relevant document."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved_ids: list, relevant_ids: list) -> float:
    """Hit Rate: 1 if at least one relevant doc in retrieved, else 0."""
    return 1.0 if any(r in retrieved_ids for r in relevant_ids) else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0

    def test_k_cutoff_respected(self):
        # relevant doc is at position 4, k=3 → missed
        assert recall_at_k(["x", "y", "z", "a"], ["a"], k=3) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=3) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        assert mean_reciprocal_rank(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0

    def test_third_position(self):
        assert mean_reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)


class TestHitRate:
    def test_hit(self):
        assert hit_rate(["a", "b"], ["a"]) == 1.0

    def test_miss(self):
        assert hit_rate(["x", "y"], ["a"]) == 0.0


class TestRRF:
    def test_basic_fusion(self):
        from api.federation import reciprocal_rank_fusion

        group_results = {
            "groupA": [
                {"chunk_id": "c1", "text": "t1", "source": "a.pdf", "page": 1, "score": 0.9},
                {"chunk_id": "c2", "text": "t2", "source": "a.pdf", "page": 2, "score": 0.7},
            ],
            "groupB": [
                {"chunk_id": "c3", "text": "t3", "source": "b.pdf", "page": 1, "score": 0.85},
                {"chunk_id": "c1", "text": "t1", "source": "a.pdf", "page": 1, "score": 0.8},
            ],
        }
        merged = reciprocal_rank_fusion(group_results)
        assert len(merged) > 0
        # c1 appare in entrambi i gruppi → dovrebbe essere primo
        assert merged[0]["chunk_id"] == "c1"

    def test_single_group(self):
        from api.federation import reciprocal_rank_fusion

        group_results = {
            "groupA": [
                {"chunk_id": "c1", "text": "t1", "source": "a.pdf", "page": 1, "score": 0.9},
            ],
        }
        merged = reciprocal_rank_fusion(group_results)
        assert len(merged) == 1


class TestNormalizeConfidence:
    def test_alta(self):
        from api.server import _normalize_confidence
        assert _normalize_confidence("Alta (Forte corrispondenza)") == "alta"

    def test_bassa(self):
        from api.server import _normalize_confidence
        assert _normalize_confidence("Bassa (Astensione forzata)") == "bassa"

    def test_media(self):
        from api.server import _normalize_confidence
        assert _normalize_confidence("Media (Corrispondenza parziale)") == "media"
