"""Postgres + pgvector backend for MemPalace.

Implements the RFC 001 BaseBackend contract using a single shared table
scoped by (palace_id, collection_name). Designed to replace the embedded
ChromaDB backend for palaces that outgrow HNSW's single-collection limits.

Configuration via environment:
    MEMPALACE_PG_DSN       — postgres connection string (required)
    MEMPALACE_PG_POOL_MAX  — max pool size (default: 10)
    MEMPALACE_PG_EMBED_DIM — embedding vector dimension (default: 384)
    MEMPALACE_PG_BATCH     — buffer size before auto-flush (default: 256)
    MEMPALACE_PG_BATCH_TTL — seconds before auto-flush even if buffer not full (default: 5)

Write path is buffered: ``add`` / ``upsert`` enqueue to an in-memory buffer
per collection and flush in bulk (one batched embedding call + one batched
INSERT). Reads (``get`` / ``query`` / ``count``) and ``delete`` always flush
first so the caller sees a consistent view. The buffer is flushed on
``close()`` and via ``atexit`` so nothing is lost on clean shutdown.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
import weakref
from typing import Any, Optional

from mempalace.backends.base import (
    BaseBackend,
    BaseCollection,
    BackendClosedError,
    GetResult,
    HealthStatus,
    PalaceRef,
    QueryResult,
    UnsupportedFilterError,
    _IncludeSpec,
)

try:
    import psycopg  # noqa: F401
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as e:
    raise ImportError(
        "mempalace-postgres requires psycopg[binary,pool]>=3.2 — install with "
        "'pip install psycopg[binary,pool]'"
    ) from e

try:
    from pgvector.psycopg import register_vector  # noqa: F401
except ImportError as e:
    raise ImportError(
        "mempalace-postgres requires pgvector>=0.3 — install with 'pip install pgvector'"
    ) from e


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains", "$gt", "$gte", "$lt", "$lte"}
)
_DEFAULT_EMBED_DIM = int(os.environ.get("MEMPALACE_PG_EMBED_DIM", "384"))
_DEFAULT_BATCH = int(os.environ.get("MEMPALACE_PG_BATCH", "256"))
_DEFAULT_BATCH_TTL = float(os.environ.get("MEMPALACE_PG_BATCH_TTL", "5"))


# Track every live collection for the atexit flush hook. WeakSet so
# collections that are garbage-collected don't keep us from exiting.
_LIVE_COLLECTIONS: "weakref.WeakSet[PostgresCollection]" = weakref.WeakSet()
_ATEXIT_REGISTERED = False
_ATEXIT_LOCK = threading.Lock()


def _register_atexit_once() -> None:
    global _ATEXIT_REGISTERED
    if _ATEXIT_REGISTERED:
        return
    with _ATEXIT_LOCK:
        if _ATEXIT_REGISTERED:
            return
        atexit.register(_flush_all_collections)
        _ATEXIT_REGISTERED = True


def _flush_all_collections() -> None:
    """Flush every live PostgresCollection on interpreter shutdown.

    Best-effort: swallows every error because atexit handlers must not raise.
    """
    for col in list(_LIVE_COLLECTIONS):
        try:
            col.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Embedder — default all-MiniLM-L6-v2 (384d) via chromadb's ONNX bundle.
# ---------------------------------------------------------------------------


_EMBED_MODEL = os.environ.get(
    "MEMPALACE_PG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
_EMBED_DEVICE = os.environ.get("MEMPALACE_PG_EMBED_DEVICE", "auto")
_EMBEDDER_KIND = os.environ.get("MEMPALACE_PG_EMBEDDER", "auto")


def _pick_device(preference: str) -> str:
    """Resolve device string for sentence-transformers.

    ``auto`` prefers CUDA when available, otherwise CPU. MPS is **not**
    picked automatically: for small embedders like all-MiniLM-L6-v2 the
    MPS kernel launch overhead makes it slower than PyTorch CPU on Apple
    Silicon. Users who want MPS for a bigger model can request it
    explicitly via ``MEMPALACE_PG_EMBED_DEVICE=mps``.

    Explicit values (``cpu`` / ``mps`` / ``cuda``) pass through after a
    feasibility check.
    """
    pref = (preference or "auto").lower()
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref == "cpu":
        return "cpu"
    mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    cuda_ok = torch.cuda.is_available()
    if pref == "mps":
        return "mps" if mps_ok else "cpu"
    if pref == "cuda":
        return "cuda" if cuda_ok else "cpu"
    # auto: prefer CUDA (real speedup), else CPU — skip MPS for small models.
    if cuda_ok:
        return "cuda"
    return "cpu"


class _DefaultEmbedder:
    """Lazy-init embedder.

    Two implementations, selected by ``MEMPALACE_PG_EMBEDDER``:

    * ``chroma-onnx`` — chromadb's bundled ``all-MiniLM-L6-v2`` ONNX model
      (CPU only). Zero extra install; dimensionally compatible with palaces
      already filed through the Chroma backend.
    * ``sentence-transformers`` — ``sentence-transformers`` with a torch
      backend. Runs on CPU / CUDA / **Apple Silicon MPS** depending on
      ``MEMPALACE_PG_EMBED_DEVICE``. Much faster on modern Macs and any
      machine with a GPU.

    Default (``auto``) picks sentence-transformers when it is importable,
    otherwise falls back to ONNX — so the Chroma fallback keeps working
    without forcing a heavy dependency.
    """

    _fn = None
    _kind = None
    _lock = threading.Lock()

    @classmethod
    def _init(cls) -> None:
        kind = _EMBEDDER_KIND.lower()
        if kind in ("auto", "sentence-transformers", "sentence_transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                device = _pick_device(_EMBED_DEVICE)
                model = SentenceTransformer(_EMBED_MODEL, device=device)
                cls._fn = lambda docs: model.encode(
                    docs, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False
                )
                cls._kind = f"sentence-transformers({device})"
                return
            except ImportError:
                if kind != "auto":
                    raise
                # fall through to ONNX
        # chroma-onnx path
        import chromadb.utils.embedding_functions as ef

        cls._fn = ef.DefaultEmbeddingFunction()
        cls._kind = "chroma-onnx"

    @classmethod
    def embed(cls, docs: list[str]) -> list[list[float]]:
        if cls._fn is None:
            with cls._lock:
                if cls._fn is None:
                    cls._init()
        if not docs:
            return []
        raw = cls._fn(docs)
        return [list(map(float, v)) for v in raw]

    @classmethod
    def kind(cls) -> str:
        if cls._fn is None:
            with cls._lock:
                if cls._fn is None:
                    cls._init()
        return cls._kind or "unknown"


# ---------------------------------------------------------------------------
# Where-clause translation: Chroma dict filter → SQL on metadata JSONB.
# ---------------------------------------------------------------------------


def _validate_where(where: Optional[dict]) -> None:
    if not where:
        return
    stack = [where]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for k, v in node.items():
            if k.startswith("$") and k not in _SUPPORTED_OPERATORS:
                raise UnsupportedFilterError(f"operator {k!r} not supported by postgres backend")
            if isinstance(v, dict):
                stack.append(v)
            elif isinstance(v, list):
                stack.extend(x for x in v if isinstance(x, dict))


def _where_to_sql(where: Optional[dict], params: list) -> str:
    if not where:
        return "TRUE"
    return _compile_node(where, params)


def _compile_node(node: Any, params: list) -> str:
    if not isinstance(node, dict):
        raise UnsupportedFilterError(f"unexpected where node: {node!r}")
    parts: list[str] = []
    for key, value in node.items():
        if key == "$and":
            if not isinstance(value, list) or not value:
                raise UnsupportedFilterError("$and requires a non-empty list")
            parts.append("(" + " AND ".join(_compile_node(x, params) for x in value) + ")")
        elif key == "$or":
            if not isinstance(value, list) or not value:
                raise UnsupportedFilterError("$or requires a non-empty list")
            parts.append("(" + " OR ".join(_compile_node(x, params) for x in value) + ")")
        elif key.startswith("$"):
            raise UnsupportedFilterError(f"top-level operator {key!r} requires field context")
        else:
            parts.append(_field_condition(key, value, params))
    return " AND ".join(parts) if parts else "TRUE"


def _field_condition(field: str, value: Any, params: list) -> str:
    col = "(metadata ->> %s)"
    if not isinstance(value, dict):
        params.append(field)
        params.append(str(value))
        return f"{col} = %s"
    if len(value) != 1:
        sub = [_field_condition(field, {k: v}, params) for k, v in value.items()]
        return "(" + " AND ".join(sub) + ")"
    op, v = next(iter(value.items()))
    if op == "$eq":
        params.append(field)
        params.append(str(v))
        return f"{col} = %s"
    if op == "$ne":
        params.append(field)
        params.append(str(v))
        return f"{col} <> %s"
    if op == "$in":
        if not v:
            return "FALSE"
        params.append(field)
        placeholders = ",".join(["%s"] * len(v))
        params.extend(str(x) for x in v)
        return f"{col} IN ({placeholders})"
    if op == "$nin":
        if not v:
            return "TRUE"
        params.append(field)
        placeholders = ",".join(["%s"] * len(v))
        params.extend(str(x) for x in v)
        return f"{col} NOT IN ({placeholders})"
    if op in ("$gt", "$gte", "$lt", "$lte"):
        sql_op = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[op]
        params.append(field)
        params.append(v)
        return f"({col})::numeric {sql_op} %s"
    if op == "$contains":
        params.append(field)
        params.append(f"%{v}%")
        return f"{col} ILIKE %s"
    raise UnsupportedFilterError(f"operator {op!r} not supported")


def _where_document_to_sql(wd: Optional[dict], params: list) -> str:
    if not wd:
        return "TRUE"
    if set(wd.keys()) - {"$contains"}:
        raise UnsupportedFilterError(f"where_document supports $contains only, got {sorted(wd)}")
    if "$contains" in wd:
        params.append(f"%{wd['$contains']}%")
        return "document ILIKE %s"
    return "TRUE"


def _palace_key(palace: PalaceRef) -> str:
    return palace.local_path or palace.id


def _vector_literal(v: list[float]) -> str:
    """Render a float list as a pgvector literal: ``[1.0,2.0,3.0]``.

    Using a string + explicit ``::vector`` cast in SQL avoids psycopg's
    default list→array adaptation (``double precision[]``) for which no
    distance operator is defined.
    """
    return "[" + ",".join(format(float(x), ".8g") for x in v) + "]"


def _sanitize_collection(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid collection name: {name!r}")
    return name


def _normalize_get_collection_args(args, kwargs) -> tuple[PalaceRef, str]:
    """Accept both kwargs-only (RFC 001) and legacy positional path signatures."""
    if "palace" in kwargs:
        palace_ref = kwargs.pop("palace")
        if not isinstance(palace_ref, PalaceRef):
            raise TypeError("palace= must be a PalaceRef instance")
        collection_name = kwargs.pop("collection_name")
        kwargs.pop("create", None)
        kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        if args:
            raise TypeError("positional args not allowed with palace= kwarg")
        return palace_ref, collection_name
    if args:
        palace_path = args[0]
        rest = list(args[1:])
        collection_name = kwargs.pop("collection_name", None) or (rest.pop(0) if rest else None)
        if collection_name is None:
            raise TypeError("collection_name is required")
        kwargs.pop("create", None)
        kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=str(palace_path), local_path=str(palace_path)),
            collection_name,
        )
    if "palace_path" in kwargs:
        palace_path = kwargs.pop("palace_path")
        collection_name = kwargs.pop("collection_name")
        kwargs.pop("create", None)
        kwargs.pop("options", None)
        if kwargs:
            raise TypeError(f"unexpected kwargs: {sorted(kwargs)}")
        return (
            PalaceRef(id=str(palace_path), local_path=str(palace_path)),
            collection_name,
        )
    raise TypeError("get_collection requires palace= or a positional palace_path")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class PostgresCollection(BaseCollection):
    """Buffered per-collection adapter.

    Writes enqueue into ``self._buffer`` and are flushed in bulk — one
    batched embedding call plus one ``executemany`` — when the buffer
    crosses ``MEMPALACE_PG_BATCH`` entries, when ``MEMPALACE_PG_BATCH_TTL``
    seconds elapse, when ``delete`` / read operations need a consistent
    view, or at process exit.
    """

    def __init__(self, pool: ConnectionPool, palace_key: str, collection_name: str):
        self._pool = pool
        self._palace = palace_key
        self._coll = _sanitize_collection(collection_name)
        # Buffered rows: list of dicts with keys: mode, id, doc, metadata, embedding
        self._buffer: list[dict] = []
        self._buffer_lock = threading.Lock()
        self._last_flush = time.monotonic()
        _LIVE_COLLECTIONS.add(self)
        _register_atexit_once()

    # ---------------- writes (buffered) ----------------

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        self._enqueue("add", documents, ids, metadatas, embeddings)

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        self._enqueue("upsert", documents, ids, metadatas, embeddings)

    def _enqueue(self, mode, documents, ids, metadatas, embeddings):
        n = len(ids)
        if len(documents) != n:
            raise ValueError("documents and ids length mismatch")
        if metadatas is not None and len(metadatas) != n:
            raise ValueError("metadatas length mismatch")
        if embeddings is not None and len(embeddings) != n:
            raise ValueError("embeddings length mismatch")

        with self._buffer_lock:
            for i in range(n):
                self._buffer.append(
                    {
                        "mode": mode,
                        "id": ids[i],
                        "doc": documents[i],
                        "metadata": metadatas[i] if metadatas is not None else {},
                        "embedding": embeddings[i] if embeddings is not None else None,
                    }
                )
            should_flush = (
                len(self._buffer) >= _DEFAULT_BATCH
                or (time.monotonic() - self._last_flush) >= _DEFAULT_BATCH_TTL
            )
            if should_flush:
                self._flush_locked()

    def flush(self) -> None:
        """Public flush — called by tests, ``close()``, and the atexit hook."""
        with self._buffer_lock:
            self._flush_locked()

    # ---------------- flush (holds the lock) ----------------

    def _flush_locked(self) -> None:
        if not self._buffer:
            self._last_flush = time.monotonic()
            return

        buf = self._buffer
        self._buffer = []
        self._last_flush = time.monotonic()

        # Collect docs that need embedding and do a single batched call.
        to_embed_idx = [i for i, row in enumerate(buf) if row["embedding"] is None]
        if to_embed_idx:
            docs_to_embed = [buf[i]["doc"] for i in to_embed_idx]
            vectors = _DefaultEmbedder.embed(docs_to_embed)
            for j, i in enumerate(to_embed_idx):
                buf[i]["embedding"] = vectors[j]

        adds = [r for r in buf if r["mode"] == "add"]
        upserts = [r for r in buf if r["mode"] == "upsert"]

        if adds:
            self._execute_batch(adds, on_conflict_update=False)
        if upserts:
            self._execute_batch(upserts, on_conflict_update=True)

    def _execute_batch(self, rows: list[dict], on_conflict_update: bool) -> None:
        if on_conflict_update:
            conflict = (
                "ON CONFLICT (palace_id, collection_name, doc_id) DO UPDATE SET "
                "document = EXCLUDED.document, metadata = EXCLUDED.metadata, "
                "embedding = EXCLUDED.embedding"
            )
        else:
            conflict = "ON CONFLICT (palace_id, collection_name, doc_id) DO NOTHING"
        sql = (
            "INSERT INTO mempalace_drawers "
            "(palace_id, collection_name, doc_id, document, metadata, embedding) "
            "VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector) " + conflict
        )
        params = [
            (
                self._palace,
                self._coll,
                r["id"],
                r["doc"],
                json.dumps(r["metadata"] or {}),
                _vector_literal(r["embedding"]),
            )
            for r in rows
        ]
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params)

    # ---------------- reads (flush first) ----------------

    def query(
        self,
        *,
        query_texts=None,
        query_embeddings=None,
        n_results=10,
        where=None,
        where_document=None,
        include=None,
    ) -> QueryResult:
        self.flush()
        _validate_where(where)
        _validate_where(where_document)
        if (query_texts is None) == (query_embeddings is None):
            raise ValueError("query requires exactly one of query_texts or query_embeddings")
        if query_texts is not None:
            if not query_texts:
                raise ValueError("query_texts must be non-empty")
            query_embeddings = _DefaultEmbedder.embed(query_texts)
        if not query_embeddings:
            raise ValueError("query_embeddings must be non-empty")

        spec = _IncludeSpec.resolve(include, default_distances=True)

        ids_all: list[list[str]] = []
        docs_all: list[list[str]] = []
        metas_all: list[list[dict]] = []
        dists_all: list[list[float]] = []
        embs_all: list[list[list[float]]] = []

        select_parts = ["doc_id"]
        if spec.documents:
            select_parts.append("document")
        if spec.metadatas:
            select_parts.append("metadata")
        select_parts.append("embedding <=> %s::vector AS distance")
        if spec.embeddings:
            select_parts.append("embedding")
        select_sql = ", ".join(select_parts)

        for qe in query_embeddings:
            qe_str = _vector_literal(qe)
            params: list = [qe_str]
            base_params = [self._palace, self._coll]
            where_sql = _where_to_sql(where, base_params)
            wd_sql = _where_document_to_sql(where_document, base_params)
            sql = (
                f"SELECT {select_sql} FROM mempalace_drawers "
                f"WHERE palace_id = %s AND collection_name = %s "
                f"AND {where_sql} AND {wd_sql} "
                f"ORDER BY embedding <=> %s::vector LIMIT %s"
            )
            params.extend(base_params)
            params.append(qe_str)
            params.append(n_results)
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()

            ids_all.append([r["doc_id"] for r in rows])
            docs_all.append([r["document"] for r in rows] if spec.documents else [])
            metas_all.append(
                [
                    r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"])
                    for r in rows
                ]
                if spec.metadatas
                else []
            )
            dists_all.append([float(r["distance"]) for r in rows] if spec.distances else [])
            if spec.embeddings:
                embs_all.append([list(r["embedding"]) for r in rows])

        return QueryResult(
            ids=ids_all,
            documents=docs_all,
            metadatas=metas_all,
            distances=dists_all,
            embeddings=embs_all if spec.embeddings else None,
        )

    def get(
        self,
        *,
        ids=None,
        where=None,
        where_document=None,
        limit=None,
        offset=None,
        include=None,
    ) -> GetResult:
        self.flush()
        _validate_where(where)
        _validate_where(where_document)
        spec = _IncludeSpec.resolve(include, default_distances=False)

        select_parts = ["doc_id"]
        if spec.documents:
            select_parts.append("document")
        if spec.metadatas:
            select_parts.append("metadata")
        if spec.embeddings:
            select_parts.append("embedding")
        select_sql = ", ".join(select_parts)

        params: list = [self._palace, self._coll]
        extra_filters: list[str] = []
        if ids is not None:
            if not ids:
                return GetResult.empty()
            placeholders = ",".join(["%s"] * len(ids))
            extra_filters.append(f"doc_id IN ({placeholders})")
            params.extend(ids)
        if where:
            extra_filters.append(_where_to_sql(where, params))
        if where_document:
            extra_filters.append(_where_document_to_sql(where_document, params))
        where_clause = " AND " + " AND ".join(extra_filters) if extra_filters else ""

        sql = (
            f"SELECT {select_sql} FROM mempalace_drawers "
            f"WHERE palace_id = %s AND collection_name = %s{where_clause} "
            f"ORDER BY doc_id"
        )
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        if offset is not None:
            sql += f" OFFSET {int(offset)}"

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        out_ids = [r["doc_id"] for r in rows]
        out_docs = [r["document"] for r in rows] if spec.documents else []
        out_metas = (
            [
                r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"])
                for r in rows
            ]
            if spec.metadatas
            else []
        )
        out_embs = [list(r["embedding"]) for r in rows] if spec.embeddings else None

        return GetResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            embeddings=out_embs,
        )

    def delete(self, *, ids=None, where=None):
        self.flush()
        _validate_where(where)
        params: list = [self._palace, self._coll]
        filters: list[str] = []
        if ids is not None:
            if not ids:
                return
            placeholders = ",".join(["%s"] * len(ids))
            filters.append(f"doc_id IN ({placeholders})")
            params.extend(ids)
        if where:
            filters.append(_where_to_sql(where, params))
        if not filters:
            raise ValueError("delete requires ids or where")
        sql = (
            "DELETE FROM mempalace_drawers "
            "WHERE palace_id = %s AND collection_name = %s AND " + " AND ".join(filters)
        )
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

    def count(self) -> int:
        self.flush()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM mempalace_drawers "
                    "WHERE palace_id = %s AND collection_name = %s",
                    (self._palace, self._coll),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                _LIVE_COLLECTIONS.discard(self)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


_SKIP_HNSW = os.environ.get("MEMPALACE_PG_SKIP_HNSW", "").lower() in ("1", "true", "yes")

_SCHEMA_SQL_CORE = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS mempalace_drawers (
    palace_id TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding vector({_DEFAULT_EMBED_DIM}),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (palace_id, collection_name, doc_id)
);

CREATE INDEX IF NOT EXISTS mempalace_drawers_palace_idx
    ON mempalace_drawers (palace_id, collection_name);

CREATE INDEX IF NOT EXISTS mempalace_drawers_metadata_idx
    ON mempalace_drawers USING GIN (metadata);
"""

_SCHEMA_SQL_HNSW = """
CREATE INDEX IF NOT EXISTS mempalace_drawers_embedding_idx
    ON mempalace_drawers USING hnsw (embedding vector_cosine_ops);
"""

_SCHEMA_SQL = _SCHEMA_SQL_CORE + ("" if _SKIP_HNSW else _SCHEMA_SQL_HNSW)


class PostgresBackend(BaseBackend):
    """MemPalace storage on Postgres + pgvector.

    A single shared table holds all drawers across every palace and
    collection, scoped by ``(palace_id, collection_name)``. This avoids
    the single-collection HNSW scale wall seen in embedded ChromaDB.
    """

    name = "postgres"
    capabilities = frozenset(
        {
            "supports_embeddings_in",
            "supports_embeddings_passthrough",
            "supports_embeddings_out",
            "supports_metadata_filters",
            "supports_contains_fast",
        }
    )

    def __init__(self):
        dsn = os.environ.get("MEMPALACE_PG_DSN") or os.environ.get("MEMPALACE_POSTGRES_DSN")
        if not dsn:
            raise RuntimeError(
                "PostgresBackend requires MEMPALACE_PG_DSN env var "
                "(e.g., postgres://user:pass@host:5432/db)"
            )
        max_size = int(os.environ.get("MEMPALACE_PG_POOL_MAX", "10"))
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=max_size,
            open=True,
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._closed = False
        # Track collections we've issued so ``close()`` can flush them.
        self._collections: list[weakref.ref[PostgresCollection]] = []
        self._collections_lock = threading.Lock()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(_SCHEMA_SQL)
            self._schema_ready = True

    def get_collection(self, *args, **kwargs) -> PostgresCollection:
        """Obtain a collection for a palace.

        Accepts both the RFC 001 kwargs-only form and the legacy positional
        form used by ``mempalace.palace.get_collection``:

        * New: ``get_collection(palace=PalaceRef, collection_name=..., create=False)``
        * Legacy: ``get_collection(palace_path, collection_name=..., create=...)``
        """
        if self._closed:
            raise BackendClosedError("PostgresBackend has been closed")
        palace_ref, collection_name = _normalize_get_collection_args(args, kwargs)
        self._ensure_schema()
        coll = PostgresCollection(self._pool, _palace_key(palace_ref), collection_name)
        with self._collections_lock:
            self._collections.append(weakref.ref(coll))
        return coll

    def close_palace(self, palace) -> None:
        # Stateless — no per-palace handles to evict.
        return None

    def close(self) -> None:
        if self._closed:
            return
        # Flush every live collection before tearing down the pool.
        with self._collections_lock:
            refs = list(self._collections)
            self._collections.clear()
        for ref in refs:
            col = ref()
            if col is not None:
                try:
                    col.flush()
                except Exception:
                    pass
        try:
            self._pool.close()
        finally:
            self._closed = True

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        if self._closed:
            return HealthStatus.unhealthy("backend closed")
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
            return HealthStatus.healthy("postgres reachable")
        except Exception as e:  # noqa: BLE001
            return HealthStatus.unhealthy(f"postgres unreachable: {e}")

    @classmethod
    def detect(cls, path: str) -> bool:
        # We never auto-detect from a local path — explicit selection only.
        return False
