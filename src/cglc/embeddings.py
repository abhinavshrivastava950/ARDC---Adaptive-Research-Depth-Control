"""Embedding backends for the dense half of hybrid retrieval (optional).

Architecture: Sec 3.3 -- the worker owns retrieval, the controller only ever
sees chunk ids. An embedder therefore lives entirely inside the worker's
retriever: swapping it changes *which* chunks come back, never their ids
(``<doc>#c<i>``), so the CGDP Unique Passage Rate (Sec 14.2) and the rest of
the controller are unaffected.

The core install stays standard-library only. Learned-model backends
(``fastembed``, ``sentence-transformers``) are imported lazily and are
optional (``pip install cglc-worker[embed]``). Without one, the zero-dependency
``HashedNgramEmbedder`` is available, but it is LEXICAL, not semantic: it
tolerates morphology and typos, it does not know that "income split" and
"revenue share" mean the same thing. ``Embedder.semantic`` says which is which,
and ``get_embedder`` reports every fallback in its note instead of hiding it.
"""
from __future__ import annotations

import math
import os
import zlib
from abc import ABC, abstractmethod
from array import array
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
BACKENDS = ("hashed", "fastembed", "sentence-transformers")
_ALIASES = {"hash": "hashed", "hashed-ngram": "hashed", "fast-embed": "fastembed",
            "sentence_transformers": "sentence-transformers", "sbert": "sentence-transformers",
            "st": "sentence-transformers"}

# A vector is a sparse {bucket: weight} dict (hashing) or a dense array('f').
Vector = Union[Dict[int, float], "array[float]"]


class EmbedderUnavailable(RuntimeError):
    """The backend cannot be used here (package missing, model failed to load)."""


def _dot(a: Vector, b: Vector) -> float:
    if isinstance(a, dict) and isinstance(b, dict):
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b[k] for k, w in a.items() if k in b)
    if isinstance(a, dict) or isinstance(b, dict):
        raise TypeError("cannot compare a sparse vector with a dense one")
    sumprod = getattr(math, "sumprod", None)  # Python >= 3.12, runs in C
    if sumprod is not None:
        return float(sumprod(a, b))
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(v: Vector) -> float:
    return math.sqrt(_dot(v, v))


class Embedder(ABC):
    """Text -> vector. ``similarity`` is cosine; backends that return unit
    vectors override it with the cheaper dot product."""

    backend: str = ""
    model: str = ""
    #: True only for real learned models that can match paraphrases/synonyms.
    semantic: bool = False
    #: Cosine floor below which a dense hit is considered irrelevant. It is a
    #: property of the model's score distribution, so each backend sets its own.
    default_min_similarity: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.backend}:{self.model}"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        """Vectors for passages (document side)."""

    @abstractmethod
    def embed_query(self, text: str) -> Vector:
        """Vector for a search query (query side; may differ from the document side)."""

    def similarity(self, qvec: Vector, dvec: Vector) -> float:
        nq, nd = _norm(qvec), _norm(dvec)
        if nq == 0.0 or nd == 0.0:
            return 0.0
        return _dot(qvec, dvec) / (nq * nd)


# --- zero-dependency lexical embedder -------------------------------------
class HashedNgramEmbedder(Embedder):
    """Feature hashing of word uni/bi-grams plus character 3-5-grams.

    TF-weighted (1 + log tf), L2-normalised, deterministic across processes
    (crc32, not Python's salted ``hash``). Lexical only: it makes the dense
    ranking robust to plurals, inflections and typos ("payments" ~ "payment",
    "recieve" ~ "receive"), but unrelated words with different spellings are
    unrelated to it. It is NOT a semantic model and says so.
    """

    backend = "hashed"
    semantic = False
    default_min_similarity = 0.15

    def __init__(self, dim: int = 1 << 20, char_ngrams: Tuple[int, ...] = (3, 4, 5),
                 word_weight: float = 1.0, bigram_weight: float = 1.0,
                 char_weight: float = 0.35) -> None:
        self.dim = int(dim)
        self.char_ngrams = tuple(char_ngrams)
        self.word_weight, self.bigram_weight, self.char_weight = word_weight, bigram_weight, char_weight
        self.model = f"hashed-ngram-v1(dim={self.dim},char={min(self.char_ngrams)}-{max(self.char_ngrams)})"

    @property
    def name(self) -> str:
        return "hashed n-gram (lexical, NOT semantic)"

    def _features(self, text: str) -> Counter:
        # One tokenizer for both sides of the hybrid: BM25's own (lowercase
        # alphanumerics, stopwords dropped). Imported lazily: retrieval.py
        # imports this module at load time.
        from .retrieval import tokenize
        words = tokenize(text)
        f: Counter = Counter()
        for w in words:
            f[f"w|{w}"] += 1
            padded = f"<{w}>"
            for n in self.char_ngrams:
                for i in range(len(padded) - n + 1):
                    f[f"c|{padded[i:i + n]}"] += 1
        for a, b in zip(words, words[1:]):
            f[f"b|{a} {b}"] += 1
        return f

    def _vector(self, text: str) -> Dict[int, float]:
        vec: Dict[int, float] = {}
        for feat, tf in self._features(text).items():
            kind = feat[0]
            weight = (self.word_weight if kind == "w" else
                      self.bigram_weight if kind == "b" else self.char_weight)
            idx = zlib.crc32(feat.encode("utf-8")) % self.dim
            vec[idx] = vec.get(idx, 0.0) + weight * (1.0 + math.log(tf))
        n = math.sqrt(sum(x * x for x in vec.values()))
        return {k: x / n for k, x in vec.items()} if n else {}

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    def similarity(self, qvec: Vector, dvec: Vector) -> float:
        return _dot(qvec, dvec)  # both unit length (or empty)


# --- learned-model backends (lazy, optional) ------------------------------
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _prefixes(model: str) -> Tuple[str, str]:
    """(query_prefix, passage_prefix) documented for common model families."""
    m = model.lower()
    if "bge" in m and "m3" not in m:
        return BGE_QUERY_INSTRUCTION, ""
    if "e5" in m:
        return "query: ", "passage: "
    return "", ""


def _min_similarity_for(model: str) -> float:
    """Cosine floor per model family. bge-small-en-v1.5 scores unrelated text
    around 0.4-0.6 (its range is compressed), so 0.55 was chosen from a small
    synthetic check (examples/retrieval_check.py --sims: all right chunks
    >= 0.58, off-topic queries <= 0.51, ~77% of wrong chunks excluded). That is a
    weak calibration. Other families are uncalibrated guesses from their usual
    score ranges; pass ``min_similarity`` to ``make_index`` to override."""
    m = model.lower()
    if "bge" in m:
        return 0.55
    if "e5" in m:
        return 0.78
    return 0.30


def _unit_dense(values) -> "array[float]":
    arr = array("f", (float(x) for x in values))
    n = math.sqrt(sum(x * x for x in arr))
    if n:
        for i in range(len(arr)):
            arr[i] /= n
    return arr


class FastEmbedEmbedder(Embedder):
    """ONNX embeddings through ``fastembed`` (no torch). Uses the library's own
    ``passage_embed`` / ``query_embed`` so model-specific prefixes are applied."""

    backend = "fastembed"
    semantic = True

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.default_min_similarity = _min_similarity_for(self.model)
        try:
            from fastembed import TextEmbedding  # type: ignore
        except ImportError as e:
            raise EmbedderUnavailable(f"fastembed is not installed ({e})") from e
        try:
            self._m = TextEmbedding(model_name=self.model)
        except Exception as e:  # download/offline/unsupported model name
            raise EmbedderUnavailable(
                f"fastembed could not load model {self.model!r} ({type(e).__name__}: {e})") from e

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        texts = list(texts)
        if not texts:
            return []
        fn = getattr(self._m, "passage_embed", None) or self._m.embed
        return [_unit_dense(v) for v in fn(texts)]

    def embed_query(self, text: str) -> Vector:
        query_embed = getattr(self._m, "query_embed", None)
        vectors = query_embed(text) if query_embed is not None else self._m.embed([text])
        for v in vectors:
            return _unit_dense(v)
        return array("f")

    def similarity(self, qvec: Vector, dvec: Vector) -> float:
        return _dot(qvec, dvec)


class SentenceTransformerEmbedder(Embedder):
    """``sentence-transformers`` (pulls in torch). Optional alternative to fastembed."""

    backend = "sentence-transformers"
    semantic = True

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.default_min_similarity = _min_similarity_for(self.model)
        self._qp, self._pp = _prefixes(self.model)
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise EmbedderUnavailable(f"sentence-transformers is not installed ({e})") from e
        try:
            self._m = SentenceTransformer(self.model)
        except Exception as e:
            raise EmbedderUnavailable(
                f"sentence-transformers could not load model {self.model!r} "
                f"({type(e).__name__}: {e})") from e

    def _encode(self, texts: List[str]) -> List[Vector]:
        arr = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [_unit_dense(v) for v in arr]

    def embed_documents(self, texts: Sequence[str]) -> List[Vector]:
        texts = [self._pp + t for t in texts]
        return self._encode(texts) if texts else []

    def embed_query(self, text: str) -> Vector:
        return self._encode([self._qp + text])[0]

    def similarity(self, qvec: Vector, dvec: Vector) -> float:
        return _dot(qvec, dvec)


# --- selection ------------------------------------------------------------
_CACHE: Dict[Tuple[str, str], Embedder] = {}     # loaded models are expensive
_FAILED: Dict[Tuple[str, str], str] = {}         # do not retry a failed download each call


def _build(backend: str, model: Optional[str]) -> Embedder:
    key = (backend, model or "")
    if key in _CACHE:
        return _CACHE[key]
    if key in _FAILED:
        raise EmbedderUnavailable(_FAILED[key])
    try:
        emb: Embedder = (HashedNgramEmbedder() if backend == "hashed" else
                         FastEmbedEmbedder(model) if backend == "fastembed" else
                         SentenceTransformerEmbedder(model))
    except EmbedderUnavailable as e:
        _FAILED[key] = str(e)
        raise
    _CACHE[key] = emb
    return emb


def get_embedder(prefer: str = "auto", model: Optional[str] = None) -> Tuple[Embedder, str]:
    """Pick an embedder and say why. Never raises, never pretends.

    ``prefer``: "auto" | "hashed" | "fastembed" | "sentence-transformers". An
    explicit non-auto value wins over the environment. With "auto", the env var
    ``CGLC_EMBEDDER`` (same values) forces a backend; otherwise the order is
    fastembed -> sentence-transformers -> hashed. ``CGLC_EMBED_MODEL`` (or
    ``model``) overrides the model of a learned backend. A forced backend that
    cannot be used falls back to hashed and the note says so.
    """
    want = (prefer or "auto").strip().lower()
    source = "prefer"
    if want == "auto":
        env = os.environ.get("CGLC_EMBEDDER", "").strip().lower()
        if env and env != "auto":
            want, source = env, "CGLC_EMBEDDER"
    want = _ALIASES.get(want, want)
    model = (model or os.environ.get("CGLC_EMBED_MODEL", "")).strip() or None

    if want not in BACKENDS and want != "auto":
        emb = _build("hashed", None)
        return emb, (f"unknown embedder {want!r} (from {source}); valid: auto, "
                     f"{', '.join(BACKENDS)}. Using {emb.name}.")

    if want == "hashed":
        emb = _build("hashed", None)
        extra = f" CGLC_EMBED_MODEL={model!r} ignored (hashed has no model)." if model else ""
        return emb, f"using {emb.name} (requested via {source}).{extra}"

    order = [want] if want != "auto" else ["fastembed", "sentence-transformers"]
    why: List[str] = []
    for backend in order:
        try:
            emb = _build(backend, model)
            return emb, f"using {emb.name}."
        except EmbedderUnavailable as e:
            why.append(f"{backend}: {e}")
    fallback = _build("hashed", None)
    head = (f"requested {want!r} (via {source}) but it is unavailable" if want != "auto"
            else "no semantic embedding backend is available")
    return fallback, (f"{head} ({'; '.join(why)}). Falling back to {fallback.name}; "
                      f"install with: pip install \"cglc-worker[embed]\".")
