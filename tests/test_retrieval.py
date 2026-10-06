"""Hybrid retrieval (BM25 + embeddings, RRF) and the RAG worker's lease enforcement.

Everything here uses scripted fakes: no network, no model download. The real
embedding backends are exercised only through fake ``fastembed`` /
``sentence_transformers`` modules injected into ``sys.modules``.
"""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from cglc import embeddings, retrieval
from cglc.contracts import TaskContract
from cglc.embeddings import (Embedder, EmbedderUnavailable, FastEmbedEmbedder,
                             HashedNgramEmbedder, SentenceTransformerEmbedder,
                             get_embedder)
from cglc.llm import LLMError, LLMReply, LLMUsage
from cglc.retrieval import BM25Index, HybridIndex, make_index, rrf_scores, tokenize
from cglc.worker.rag_worker import HYBRID_QUERY_NOTE, RAGDocumentWorker


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No test may reach a real embedding model or depend on the user's env."""
    monkeypatch.setenv("CGLC_EMBEDDER", "hashed")
    monkeypatch.delenv("CGLC_EMBED_MODEL", raising=False)
    monkeypatch.setattr(embeddings, "_CACHE", {})
    monkeypatch.setattr(embeddings, "_FAILED", {})


# --- fakes -------------------------------------------------------------------
_CONCEPT = {"income": "money", "revenue": "money", "earnings": "money",
            "split": "divide", "share": "divide", "division": "divide",
            "hacker": "intrusion", "intruder": "intrusion"}


class SynonymEmbedder(Embedder):
    """A deterministic stand-in for a learned model: bag of synonym concepts."""
    backend, model, semantic = "fake-semantic", "synonym-map-v1", True
    default_min_similarity = 0.3

    def _vec(self, text):
        counts = {}
        for t in tokenize(text):
            c = _CONCEPT.get(t, t)
            counts[c] = counts.get(c, 0) + 1
        n = sum(x * x for x in counts.values()) ** 0.5
        return {k: v / n for k, v in counts.items()} if n else {}

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def similarity(self, qvec, dvec):
        return sum(w * dvec.get(k, 0.0) for k, w in qvec.items())


class ScriptedEmbedder(Embedder):
    """Dense similarities are looked up in a table keyed by (query, chunk text)."""
    backend, model, semantic = "scripted", "table", True
    default_min_similarity = 0.3

    def __init__(self, table):
        self.table = table

    def embed_documents(self, texts):
        return list(texts)

    def embed_query(self, text):
        return text

    def similarity(self, qvec, dvec):
        return self.table.get((qvec, dvec), 0.0)


class ExplodingEmbedder(SynonymEmbedder):
    def embed_documents(self, texts):
        raise AssertionError("the corpus must not be embedded here")


DOCS = {"d.txt": "\n\n".join(
    ["Overview of the platform and its goals."]
    + ["The revenue share is 88 percent to creators and 12 percent to the platform."]
    + [f"Office plant number {i} is watered on Mondays by the facilities team." for i in range(8)]
)}
REV = "d.txt#c1"


# --- RRF ---------------------------------------------------------------------
def test_rrf_scores_math():
    s = rrf_scores([[0, 1, 2], [2, 0]], k=60)
    assert s[0] == pytest.approx(1 / 61 + 1 / 62)
    assert s[1] == pytest.approx(1 / 62)
    assert s[2] == pytest.approx(1 / 63 + 1 / 61)
    assert rrf_scores([], 60) == {}


def test_hybrid_order_matches_hand_computed_rrf_with_a_dense_cutoff():
    texts = ["zeta zeta zeta alpha", "zeta beta beta beta beta beta", "plain gamma text",
             "plain delta text", "zeta epsilon epsilon epsilon epsilon epsilon epsilon epsilon"]
    docs = {"d": "\n\n".join(texts)}
    q = "zeta"
    bm = BM25Index(docs)
    bm_order = [c.chunk_id for c in bm.search(q, 10)]
    assert set(bm_order) == {"d#c0", "d#c1", "d#c4"}
    dense = {"d#c3": 0.9, "d#c0": 0.8, "d#c2": 0.7, "d#c4": 0.1}  # c4 is under the cutoff
    emb = ScriptedEmbedder({(q, texts[int(k[-1])]): v for k, v in dense.items()})
    hy = HybridIndex(docs, emb, min_similarity=0.3)
    dense_order = ["d#c3", "d#c0", "d#c2"]
    expect = {}
    for ranking in (bm_order, dense_order):
        for r, cid in enumerate(ranking, start=1):
            expect[cid] = expect.get(cid, 0.0) + 1.0 / (60 + r)
    got = [c.chunk_id for c in hy.search(q, 10)]
    assert sorted(got) == sorted(expect)                 # c4 enters through BM25 only, not via dense
    assert [expect[c] for c in got] == sorted(expect.values(), reverse=True)
    assert hy.search(q, 2) == hy.search(q, 10)[:2]


def test_dense_hits_below_the_similarity_floor_are_not_injected():
    docs = {"d": "zeta one\n\nunrelated two\n\nunrelated three"}
    emb = ScriptedEmbedder({("zeta", "unrelated two"): 0.29, ("zeta", "unrelated three"): 0.31})
    ids = [c.chunk_id for c in HybridIndex(docs, emb).search("zeta", 5)]
    assert ids == ["d#c0", "d#c2"]      # BM25 hit, then the dense hit above 0.3; 0.29 is dropped
    ids = [c.chunk_id for c in HybridIndex(docs, emb, min_similarity=0.2).search("zeta", 5)]
    assert ids == ["d#c0", "d#c2", "d#c1"]


def test_ties_break_by_bm25_rank_then_dense_rank_then_document_order():
    docs = {"d": "dense only text\n\nzeta bm25 only"}
    hy = HybridIndex(docs, ScriptedEmbedder({("zeta", "dense only text"): 0.9}))
    # c0 (dense rank 1) and c1 (BM25 rank 1) both score 1/61: BM25's pick goes first
    assert [c.chunk_id for c in hy.search("zeta", 5)] == ["d#c1", "d#c0"]
    # identical dense scores: document order decides
    docs = {"d": "same one\n\nsame two\n\nsame three"}
    emb = ScriptedEmbedder({("q", t): 0.8 for t in ("same one", "same two", "same three")})
    first = [c.chunk_id for c in HybridIndex(docs, emb).search("q", 5)]
    assert first == ["d#c0", "d#c1", "d#c2"]
    for _ in range(3):
        assert [c.chunk_id for c in HybridIndex(docs, emb).search("q", 5)] == first


def test_exclude_is_honoured_on_both_sides_like_bm25():
    hy = HybridIndex(DOCS, SynonymEmbedder())
    bm = BM25Index(DOCS)
    assert [c.chunk_id for c in hy.search("revenue share creators", 3)][0] == REV
    out = hy.search("revenue share creators", 5, exclude=[REV])
    assert REV not in {c.chunk_id for c in out}
    assert hy.search("income split", 5, exclude=[REV]) == []     # dense-only target excluded
    assert all(c.chunk_id != REV for c in bm.search("revenue share creators", 5, exclude=[REV]))
    everything = [c.chunk_id for c in hy.chunks]
    assert hy.search("revenue share creators", 5, exclude=everything) == []


def test_empty_or_stopword_only_queries_and_zero_k_return_nothing():
    hy = HybridIndex(DOCS, SynonymEmbedder())
    assert hy.search("", 5) == [] and hy.search("the of and", 5) == []
    assert hy.search("revenue", 0) == []
    assert HybridIndex({}, SynonymEmbedder()).search("revenue", 5) == []


def test_same_interface_and_chunk_ids_as_bm25():
    bm, hy = BM25Index(DOCS), HybridIndex(DOCS, SynonymEmbedder())
    assert isinstance(hy, BM25Index)
    assert [c.chunk_id for c in hy.chunks] == [c.chunk_id for c in bm.chunks]
    assert len(hy) == len(bm) and hy.get(REV) == bm.get(REV)
    assert hy.openings() == bm.openings()
    assert hy.search("revenue", 3)[0].chunk_id == REV


def test_bm25_ordering_is_unchanged_by_the_refactor():
    idx = BM25Index({"a": "cat dog dog\n\ncat\n\ndog dog dog bird\n\nbird"})
    assert [c.chunk_id for c in idx.search("dog cat", 4)] == ["a#c0", "a#c2", "a#c1"]
    assert idx.search("zebra", 4) == [] and idx.search("the", 4) == []


# --- the point of the exercise ------------------------------------------------
def test_a_paraphrase_is_found_by_hybrid_but_not_by_bm25():
    q = "income split"
    assert BM25Index(DOCS).search(q, 3) == []
    hy = HybridIndex(DOCS, SynonymEmbedder())
    assert [c.chunk_id for c in hy.search(q, 3)] == [REV]
    row = hy.explain(q, 3)[0]
    assert row["chunk_id"] == REV and row["bm25_rank"] is None and row["dense_rank"] == 1
    assert row["similarity"] > 0.3


# --- hashed embedder -----------------------------------------------------------
def test_hashed_embedder_is_declared_lexical_not_semantic():
    e = HashedNgramEmbedder()
    assert e.semantic is False and "NOT semantic" in e.name and "hashed" in e.backend
    v = e.embed_query("payment processing")
    assert sum(x * x for x in v.values()) == pytest.approx(1.0)
    assert e.similarity(v, v) == pytest.approx(1.0)
    assert e.embed_query("the of and") == {}


def test_hashed_embedder_handles_word_forms_and_typos_where_bm25_cannot():
    docs = {"d": "Payments are processed by the billing service every night.\n\n"
                 "Gardeners water the office plants on Mondays.\n\n"
                 "Parental leave lasts sixteen weeks."}
    bm = BM25Index(docs)
    hy = HybridIndex(docs, HashedNgramEmbedder())
    for q, want in (("payment processing", "d#c0"), ("payment procesing", "d#c0"),   # inflections, a typo
                    ("gardener watering plant", "d#c1")):
        assert bm.search(q, 3) == [], q
        assert hy.search(q, 3)[0].chunk_id == want, q
    e = HashedNgramEmbedder()
    dv = e.embed_documents([docs["d"].split("\n\n")[1]])[0]
    assert e.similarity(e.embed_query("quantum chromodynamics"), dv) < e.default_min_similarity
    assert hy.search("quantum chromodynamics", 3) == []


def test_hashed_vectors_do_not_depend_on_pythons_salted_hash():
    code = ("import json;from cglc.embeddings import HashedNgramEmbedder as H;"
            "v=H().embed_query('Revenue share, 88 percent');"
            "print(json.dumps(sorted(v.items())))")
    src = str(Path(retrieval.__file__).resolve().parents[1])
    outs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": src}
        outs.append(subprocess.run([sys.executable, "-c", code], env=env, check=True,
                                   capture_output=True, text=True).stdout)
    assert outs[0] == outs[1] and json.loads(outs[0])


# --- make_index -----------------------------------------------------------------
INFO_KEYS = {"requested", "used", "dense_backend", "dense_model", "semantic", "n_chunks",
             "fusion", "rrf_k", "min_similarity", "note", "dense_depth",
             "max_dense_chunks", "bm25_k1", "bm25_b"}


def test_make_index_bm25_never_touches_an_embedder(monkeypatch):
    monkeypatch.setattr(retrieval, "get_embedder",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no embedder wanted")))
    idx, info = make_index(DOCS, "bm25", embedder=ExplodingEmbedder())
    assert type(idx) is BM25Index
    assert info["requested"] == "bm25" and info["used"] == "bm25"
    assert info["dense_backend"] is None and info["semantic"] is False and info["fusion"] is None
    assert info["n_chunks"] == len(idx) == 10 and set(info) == INFO_KEYS
    json.dumps(info)


def test_make_index_auto_uses_hybrid_only_for_a_semantic_backend():
    idx, info = make_index(DOCS, "auto", SynonymEmbedder())
    assert isinstance(idx, HybridIndex)
    assert info["requested"] == "auto" and info["used"] == "hybrid"
    assert info["dense_backend"] == "fake-semantic" and info["dense_model"] == "synonym-map-v1"
    assert info["semantic"] is True and info["fusion"] == "rrf" and info["rrf_k"] == 60
    assert info["min_similarity"] == 0.3 and info["dense_depth"] == 50
    assert info["max_dense_chunks"] == 3000 and info["n_chunks"] == 10
    assert set(info) == INFO_KEYS
    json.dumps(info)
    assert [c.chunk_id for c in idx.search("income split", 3)] == [REV]


def test_make_index_auto_with_the_hashed_fallback_keeps_plain_bm25():
    for emb in (None, HashedNgramEmbedder()):          # None -> get_embedder() -> hashed (env fixture)
        idx, info = make_index(DOCS, "auto", emb)
        assert type(idx) is BM25Index and info["used"] == "bm25"
        assert info["semantic"] is False and info["dense_backend"] is None
        assert "lexical" in info["note"] and "dilute" in info["note"]


def test_make_index_auto_resolves_the_embedder_through_get_embedder(monkeypatch):
    monkeypatch.setattr(retrieval, "get_embedder", lambda *a, **k: (SynonymEmbedder(), "picked"))
    idx, info = make_index(DOCS)
    assert info["used"] == "hybrid" and info["note"] == "picked"
    assert isinstance(idx, HybridIndex)


def test_make_index_forced_hybrid_with_hashed_is_allowed_but_flagged():
    idx, info = make_index(DOCS, "hybrid", HashedNgramEmbedder())
    assert isinstance(idx, HybridIndex) and info["used"] == "hybrid"
    assert info["semantic"] is False and info["dense_backend"] == "hashed"
    assert info["min_similarity"] == HashedNgramEmbedder.default_min_similarity
    assert "NOT" in info["note"] and "paraphrases" in info["note"]


def test_make_index_records_every_configured_value_and_accepts_overrides():
    idx, info = make_index(DOCS, "hybrid", SynonymEmbedder(), min_similarity=0.5, rrf_k=10,
                           depth=7, max_dense_chunks=99, k1=1.2, b=0.5)
    assert (info["min_similarity"], info["rrf_k"], info["dense_depth"]) == (0.5, 10, 7)
    assert (info["max_dense_chunks"], info["bm25_k1"], info["bm25_b"]) == (99, 1.2, 0.5)
    assert (idx.rrf_k, idx.depth, idx.min_similarity, idx.k1, idx.b) == (10, 7, 0.5, 1.2, 0.5)


def test_make_index_caps_the_number_of_embedded_chunks(monkeypatch):
    monkeypatch.setattr(retrieval, "MAX_DENSE_CHUNKS", 5)
    for mode in ("auto", "hybrid"):
        idx, info = make_index(DOCS, mode, ExplodingEmbedder())     # 10 chunks > 5
        assert type(idx) is BM25Index and info["used"] == "bm25"
        assert info["max_dense_chunks"] == 5 and "MAX_DENSE_CHUNKS=5" in info["note"]
        assert info["dense_backend"] is None and info["n_chunks"] == 10
    _idx, info = make_index(DOCS, "hybrid", SynonymEmbedder(), max_dense_chunks=10)
    assert info["used"] == "hybrid"                                  # exactly at the cap is fine


def test_make_index_survives_an_embedder_that_fails_while_indexing():
    class Broken(SynonymEmbedder):
        def embed_documents(self, texts):
            raise RuntimeError("onnx session died")
    idx, info = make_index(DOCS, "hybrid", Broken())
    assert type(idx) is BM25Index and info["used"] == "bm25" and info["semantic"] is False
    assert "RuntimeError" in info["note"] and "onnx session died" in info["note"]


def test_make_index_rejects_unknown_modes_and_handles_an_empty_corpus():
    with pytest.raises(ValueError):
        make_index(DOCS, "dense")
    idx, info = make_index({}, "hybrid", SynonymEmbedder())
    assert len(idx) == 0 and info["used"] == "bm25" and info["n_chunks"] == 0 and info["note"]


def test_chunk_ids_are_identical_whatever_the_retriever():
    ids = lambda i: [c.chunk_id for c in i.chunks]
    base = ids(make_index(DOCS, "bm25")[0])
    assert ids(make_index(DOCS, "hybrid", SynonymEmbedder())[0]) == base
    assert base == [f"d.txt#c{i}" for i in range(10)]


# --- get_embedder -------------------------------------------------------------------
def _fake_fastembed(monkeypatch, fail=None):
    log = {"init": [], "query": [], "passage": []}

    class TextEmbedding:
        def __init__(self, model_name=None, **kw):
            log["init"].append(model_name)
            if fail:
                raise fail

        def query_embed(self, q, **kw):
            log["query"].append(q)
            yield [3.0, 4.0]

        def passage_embed(self, texts, **kw):
            log["passage"].append(list(texts))
            for _ in log["passage"][-1]:
                yield [1.0, 0.0]

        def embed(self, *a, **kw):
            raise AssertionError("the query/passage functions must be used")

    mod = types.ModuleType("fastembed")
    mod.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", mod)
    return log


def _fake_sbert(monkeypatch):
    log = {"init": [], "encode": []}

    class SentenceTransformer:
        def __init__(self, model):
            log["init"].append(model)

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            log["encode"].append(list(texts))
            return [[0.0, 1.0] for _ in texts]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return log


def _missing(monkeypatch, *names):
    for n in names:
        monkeypatch.setitem(sys.modules, n, None)        # makes `import n` raise ImportError


def test_get_embedder_hashed_when_forced_or_nothing_is_installed(monkeypatch):
    e, note = get_embedder()                              # env fixture forces hashed
    assert isinstance(e, HashedNgramEmbedder) and not e.semantic and "CGLC_EMBEDDER" in note
    monkeypatch.delenv("CGLC_EMBEDDER")
    _missing(monkeypatch, "fastembed", "sentence_transformers")
    e, note = get_embedder()
    assert isinstance(e, HashedNgramEmbedder)
    assert "no semantic embedding backend" in note and "lexical" in note and "NOT semantic" in note
    assert "fastembed" in note and "sentence-transformers" in note


def test_a_forced_backend_that_is_not_installed_falls_back_to_hashed_with_a_note(monkeypatch):
    _missing(monkeypatch, "fastembed", "sentence_transformers")
    for forced in ("fastembed", "sentence-transformers", "sentence_transformers"):
        monkeypatch.setenv("CGLC_EMBEDDER", forced)
        e, note = get_embedder()
        assert isinstance(e, HashedNgramEmbedder) and e.semantic is False
        assert "unavailable" in note and "not installed" in note and "CGLC_EMBEDDER" in note
    e, note = get_embedder(prefer="fastembed")           # explicit argument behaves the same
    assert isinstance(e, HashedNgramEmbedder) and "not installed" in note


def test_a_forced_backend_does_not_silently_use_a_different_learned_one(monkeypatch):
    _fake_fastembed(monkeypatch)
    _missing(monkeypatch, "sentence_transformers")
    monkeypatch.setenv("CGLC_EMBEDDER", "sentence-transformers")
    e, note = get_embedder()
    assert isinstance(e, HashedNgramEmbedder) and "sentence-transformers" in note


def test_unknown_backend_name_falls_back_with_a_note(monkeypatch):
    monkeypatch.setenv("CGLC_EMBEDDER", "openai")
    e, note = get_embedder()
    assert isinstance(e, HashedNgramEmbedder) and "unknown embedder 'openai'" in note


def test_auto_prefers_fastembed_then_sentence_transformers(monkeypatch):
    monkeypatch.delenv("CGLC_EMBEDDER")
    flog = _fake_fastembed(monkeypatch)
    slog = _fake_sbert(monkeypatch)
    e, note = get_embedder()
    assert isinstance(e, FastEmbedEmbedder) and e.semantic is True
    assert e.model == "BAAI/bge-small-en-v1.5" and flog["init"] == ["BAAI/bge-small-en-v1.5"]
    assert slog["init"] == [] and "fastembed" in note
    monkeypatch.setattr(embeddings, "_CACHE", {})
    _missing(monkeypatch, "fastembed")
    e, _ = get_embedder()
    assert isinstance(e, SentenceTransformerEmbedder) and e.semantic is True


def test_fastembed_uses_the_query_and_passage_functions_and_unit_vectors(monkeypatch):
    log = _fake_fastembed(monkeypatch)
    e, _ = get_embedder(prefer="fastembed")
    q = e.embed_query("what is x")
    d = e.embed_documents(["p1", "p2"])
    assert log["query"] == ["what is x"] and log["passage"] == [["p1", "p2"]]
    assert list(q) == pytest.approx([0.6, 0.8]) and list(d[0]) == pytest.approx([1.0, 0.0])
    assert e.similarity(q, d[0]) == pytest.approx(0.6)
    assert e.embed_documents([]) == []


def test_embed_model_env_override_and_cache(monkeypatch):
    log = _fake_fastembed(monkeypatch)
    monkeypatch.setenv("CGLC_EMBEDDER", "fastembed")
    monkeypatch.setenv("CGLC_EMBED_MODEL", "intfloat/e5-small-v2")
    e, _ = get_embedder()
    assert e.model == "intfloat/e5-small-v2" and log["init"] == ["intfloat/e5-small-v2"]
    assert get_embedder()[0] is e and len(log["init"]) == 1      # loaded models are cached
    monkeypatch.setenv("CGLC_EMBEDDER", "hashed")
    _e, note = get_embedder()
    assert "ignored" in note                                      # no model for the hashed backend


def test_explicit_prefer_wins_over_the_environment(monkeypatch):
    _fake_fastembed(monkeypatch)
    monkeypatch.setenv("CGLC_EMBEDDER", "hashed")
    assert isinstance(get_embedder(prefer="fastembed")[0], FastEmbedEmbedder)
    monkeypatch.setenv("CGLC_EMBEDDER", "fastembed")
    assert isinstance(get_embedder(prefer="hashed")[0], HashedNgramEmbedder)


def test_a_model_that_fails_to_load_falls_back_and_is_not_retried(monkeypatch):
    log = _fake_fastembed(monkeypatch, fail=OSError("offline: cannot download"))
    monkeypatch.delenv("CGLC_EMBEDDER")
    _missing(monkeypatch, "sentence_transformers")
    for _ in range(2):
        e, note = get_embedder()
        assert isinstance(e, HashedNgramEmbedder)
        assert "could not load model" in note and "offline: cannot download" in note
    assert len(log["init"]) == 1


def test_sentence_transformer_backend_applies_the_documented_prefixes(monkeypatch):
    log = _fake_sbert(monkeypatch)
    e, _ = get_embedder(prefer="sentence-transformers")
    assert e.semantic and e.model == "BAAI/bge-small-en-v1.5"
    e.embed_query("how are refunds handled")
    assert log["encode"][-1] == [embeddings.BGE_QUERY_INSTRUCTION + "how are refunds handled"]
    e.embed_documents(["a passage"])
    assert log["encode"][-1] == ["a passage"]
    e5, _ = get_embedder(prefer="sentence-transformers", model="intfloat/e5-small-v2")
    e5.embed_query("q")
    assert log["encode"][-1] == ["query: q"]
    e5.embed_documents(["p"])
    assert log["encode"][-1] == ["passage: p"]


def test_constructing_an_unavailable_backend_directly_raises_a_clear_error(monkeypatch):
    _missing(monkeypatch, "fastembed")
    with pytest.raises(EmbedderUnavailable, match="not installed"):
        FastEmbedEmbedder()


# --- RAG worker: retrieval selection and lease enforcement ----------------------------
FILLER = [f"Filler section {i} about nothing in particular, widgets widgets widgets." for i in range(30)]
RAG_DOC = "\n\n".join(FILLER + ["The revenue split is 88 percent to the user and 12 percent to the platform.",
                                "Advertisers fund campaigns through Razorpay checkout."])
SPLIT_QUOTE = "The revenue split is 88 percent to the user"
SPLIT_ID = "d.txt#c30"


class ScriptedLLM:
    model = "fake"

    def __init__(self, queries=("revenue split user platform percent",), citations=(), fail_step=None):
        self.queries, self.citations, self.fail_step = list(queries), list(citations), fail_step
        self.query_calls, self.step_prompts, self.query_systems = [], [], []

    def complete_json(self, system, user, schema, max_tokens=8000, temperature=None):
        u = LLMUsage(input_tokens=10, output_tokens=5)
        if set(schema["properties"]) == {"queries"}:
            self.query_calls.append(user)
            self.query_systems.append(str(system))
            return LLMReply({"queries": self.queries}, u, 0.0)
        if self.fail_step:
            raise self.fail_step
        self.step_prompts.append(user)
        return LLMReply({"query": "checked", "citations": self.citations, "draft": "a draft",
                         "propose_final": False, "contradiction": False, "blocker": ""}, u, 0.0)

    @property
    def calls(self):
        return len(self.query_calls) + len(self.step_prompts)


def contract():
    return TaskContract.from_dict({"goal": "figure out the compensation scheme",
                                   "evidence_obligations": ["scheme quoted"]})


def make_worker(llm=None, top_k=3, **kw):
    llm = llm or ScriptedLLM()
    return RAGDocumentWorker({"d.txt": RAG_DOC}, contract(), llm, top_k=top_k, **kw), llm


def cited():
    return [{"source_id": "d.txt", "quote": SPLIT_QUOTE}]


def test_defaults_keep_plain_bm25_and_expose_the_retrieval_record():
    w, _ = make_worker()
    assert type(w.index) is BM25Index
    info = w.retrieval_info
    assert info["requested"] == "auto" and info["used"] == "bm25" and info["n_chunks"] == 32
    json.dumps(info)
    assert make_worker(retriever="bm25")[0].retrieval_info["requested"] == "bm25"


def test_a_semantic_embedder_makes_the_worker_find_a_paraphrase_bm25_misses():
    queries = ["income division creators"]
    plain, llm1 = make_worker(ScriptedLLM(queries), retriever="bm25")
    hyb, llm2 = make_worker(ScriptedLLM(queries), retriever="auto", embedder=SynonymEmbedder())
    r_plain = plain.act("CONTINUE", ["ev-0"], [], "")
    r_hyb = hyb.act("CONTINUE", ["ev-0"], [], "")
    assert SPLIT_ID not in {o.chunk_id for o in r_plain.observations}
    assert SPLIT_ID in {o.chunk_id for o in r_hyb.observations}
    assert hyb.retrieval_info["used"] == "hybrid" and hyb.retrieval_info["semantic"] is True
    assert HYBRID_QUERY_NOTE in llm2.query_systems[0] and HYBRID_QUERY_NOTE not in llm1.query_systems[0]
    assert all(o.chunk_id.startswith("d.txt#c") for o in r_hyb.observations)   # same ids as BM25


def test_a_lexical_hybrid_does_not_tell_the_model_it_matches_meaning():
    w, llm = make_worker(retriever="hybrid")           # env fixture -> hashed embedder
    assert w.retrieval_info["used"] == "hybrid" and w.retrieval_info["semantic"] is False
    w.act("CONTINUE", ["ev-0"], [], "")
    assert HYBRID_QUERY_NOTE not in llm.query_systems[0]


def test_search_allowed_runs_the_query_call_and_reports_search():
    w, llm = make_worker(ScriptedLLM(citations=cited()))
    res = w.act("CONTINUE", ["ev-0"], ["SEARCH", "READ", "ANSWER"], "")
    assert len(llm.query_calls) == 1 and len(llm.step_prompts) == 1
    assert res.detail["action_class"] == "SEARCH" and SPLIT_ID in {o.chunk_id for o in res.observations}
    assert w.allowed_classes == ["SEARCH", "READ", "ANSWER"]


def test_an_empty_allowed_list_means_unrestricted():
    w, llm = make_worker()
    res = w.act("VERIFY", ["ev-0"], [], "")
    assert w.allowed_classes == [] and res.detail["action_class"] == "SEARCH" and len(llm.query_calls) == 1


def test_without_search_and_nothing_seen_nothing_is_retrieved_or_invented():
    for allowed in (["READ", "ANSWER"], ["VERIFY"], ["READ"]):
        w, llm = make_worker()
        res = w.act("CONTINUE", ["ev-0"], allowed, "unchanged draft")
        assert llm.calls == 0                              # no query call, no step call
        assert res.observations == [] and res.draft == "unchanged draft"
        assert res.detail["action_class"] in allowed and res.detail["note"]
        assert res.detail["cost"]["tool_calls"] == 1.0 and res.detail["cost"]["tokens"] == 0.0
        assert not res.blocker and not res.propose_final
        assert w.seen_chunks == set()


def test_without_search_the_worker_rereads_cited_passages_first_and_does_not_search():
    w, llm = make_worker(ScriptedLLM(citations=cited()), top_k=2)
    w.act("CONTINUE", ["ev-0"], [], "")                    # SEARCH step: cites the split chunk
    llm.queries, llm.citations = ["widgets filler section"], []
    w.act("CONTINUE", ["ev-0"], [], "")                    # later, more recent, uncited passages
    assert SPLIT_ID in w.evidence and w._seen_order[-1] != SPLIT_ID
    before_queries = len(llm.query_calls)
    res = w.act("CONTINUE", ["ev-0"], ["READ", "ANSWER"], "a draft")
    assert len(llm.query_calls) == before_queries          # no query-writing call
    assert res.detail["action_class"] == "READ"
    ids = [o.chunk_id for o in res.observations]
    assert ids[0] == SPLIT_ID and len(ids) == 2 and set(ids) <= w.seen_chunks
    assert "NO NEW SEARCH IS PERMITTED" in llm.step_prompts[-1] and SPLIT_QUOTE in llm.step_prompts[-1]
    assert "SEARCH QUERIES USED" not in llm.step_prompts[-1]
    assert res.detail["action_text"].startswith("READ ") and res.detail["reread"] == ids


def test_reread_falls_back_to_the_most_recently_seen_passages():
    w, llm = make_worker(ScriptedLLM(), top_k=3)           # nothing cited
    w.act("CONTINUE", ["ev-0"], [], "")
    llm.queries = ["widgets filler section"]
    w.act("CONTINUE", ["ev-0"], [], "")
    order = list(w._seen_order)
    assert len(order) == 4
    res = w.act("CONTINUE", ["ev-0"], ["READ"], "a draft")
    assert [o.chunk_id for o in res.observations] == list(reversed(order))[:3]


def test_class_reported_follows_intent_and_permissions():
    def run(intent, allowed):
        w, _ = make_worker(ScriptedLLM(citations=cited()))
        w.act("CONTINUE", ["ev-0"], [], "")
        return w.act(intent, ["ev-0"], allowed, "a draft").detail["action_class"]
    assert run("VERIFY", ["READ", "VERIFY", "ANSWER"]) == "VERIFY"
    assert run("VERIFY", ["READ", "ANSWER"]) == "READ"          # VERIFY not permitted
    assert run("CONTINUE", ["READ", "VERIFY", "ANSWER"]) == "READ"
    assert run("CONTINUE", ["VERIFY", "ANSWER"]) == "VERIFY"    # READ not permitted, VERIFY is
    assert run("REDIRECT", ["READ"]) == "READ"


def test_answer_only_lease_revises_the_draft_without_retrieving_or_reading():
    w, llm = make_worker(ScriptedLLM(citations=cited()))
    w.act("CONTINUE", ["ev-0"], [], "")
    before = len(w.seen_chunks)
    res = w.act("CONTINUE", ["ev-0"], ["ANSWER"], "a draft")
    assert res.detail["action_class"] == "ANSWER" and res.observations == []
    assert "NO SEARCH AND NO NEW READING" in llm.step_prompts[-1]
    assert "<passage" not in llm.step_prompts[-1] and len(w.seen_chunks) == before
    w2, llm2 = make_worker()
    res = w2.act("CONTINUE", ["ev-0"], ["ANSWER"], "")           # nothing to revise
    assert llm2.calls == 0 and res.observations == [] and res.detail["action_class"] == "ANSWER"


def test_redirect_search_excludes_already_seen_chunks():
    llm = ScriptedLLM(queries=["widgets filler section"])
    w, _ = make_worker(llm, top_k=3)
    first = {o.chunk_id for o in w.act("CONTINUE", ["ev-0"], ["SEARCH"], "").observations}
    again = {o.chunk_id for o in w.act("CONTINUE", ["ev-0"], ["SEARCH"], "").observations}
    redirected = w.act("REDIRECT", ["ev-0"], ["SEARCH", "READ"], "")
    new = {o.chunk_id for o in redirected.observations}
    assert redirected.detail["action_class"] == "SEARCH"
    assert new and not (new & (first | again))


def test_every_result_carries_an_action_class_even_when_the_model_call_fails():
    w, _ = make_worker(ScriptedLLM(fail_step=LLMError("rate limited")))
    res = w.act("CONTINUE", ["ev-0"], ["SEARCH"], "d")
    assert res.blocker and res.detail["action_class"] == "SEARCH"
    res = w.act("CONTINUE", ["ev-0"], ["READ"], "d")
    assert res.blocker and res.detail["action_class"] == "READ"
