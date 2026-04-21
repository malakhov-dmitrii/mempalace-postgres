"""Postgres + pgvector backend for MemPalace.

Implements the RFC 001 BaseBackend contract using a single shared table
scoped by (palace_id, collection_name). Designed to replace the embedded
ChromaDB backend for palaces that outgrow HNSW's single-collection limits.

Configuration via environment:
    MEMPALACE_PG_DSN     — postgres connection string (required)
    MEMPALACE_PG_POOL_MAX — max pool size (default: 10)
    MEMPALACE_PG_EMBED_DIM — embedding vector dimension (default: 384)
"""

from __future__ import annotations

import json
import os
import re
import threading
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
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as e:
    raise ImportError(
        "mempalace-postgres requires psycopg[binary,pool]>=3.2 — install with "
        "'pip install psycopg[binary,pool]'"
    ) from e

try:
    from pgvector.psycopg import register_vector
except ImportError as e:
    raise ImportError(
        "mempalace-postgres requires pgvector>=0.3 — install with 'pip install pgvector'"
    ) from e


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUPPORTED_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$and", "$or", "$contains", "$gt", "$gte", "$lt", "$lte"}
)
_DEFAULT_EMBED_DIM = int(os.environ.get("MEMPALACE_PG_EMBED_DIM", "384"))


# ---------------------------------------------------------------------------
# Embedder — default all-MiniLM-L6-v2 (384d) via chromadb's ONNX bundle.
# ---------------------------------------------------------------------------


class _DefaultEmbedder:
    """Lazy-init default embedder using chromadb's ONNX model.

    mempalace already depends on chromadb, so we piggyback on its default
    embedding function instead of adding sentence-transformers. Produces
    384-dimensional vectors compatible with the ``vector(384)`` column.
    """

    _fn = None
    _lock = threading.Lock()

    @classmethod
    def embed(cls, docs: list[str]) -> list[list[float]]:
        if cls._fn is None:
            with cls._lock:
                if cls._fn is None:
                    import chromadb.utils.embedding_functions as ef

                    cls._fn = ef.DefaultEmbeddingFunction()
        raw = cls._fn(docs)
        return [list(map(float, v)) for v in raw]


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


# ---------------------------------------------------------------------------
# Palace identity — stable key across PalaceRef fields.
# ---------------------------------------------------------------------------


def _palace_key(palace: PalaceRef) -> str:
    return palace.local_path or palace.id


def _vector_literal(v: list[float]) -> str:
    """Render a float list as a pgvector literal: ``[1.0,2.0,3.0]``.

    Using a string + explicit ``::vector`` cast in SQL avoids psycopg's
    default list→array adaptation, which yields ``double precision[]``
    for which no distance operator is defined.
    """
    return "[" + ",".join(format(float(x), ".8g") for x in v) + "]"


def _sanitize_collection(name: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"invalid collection name: {name!r}")
    return name


def _normalize_get_collection_args(args, kwargs) -> tuple[PalaceRef, str]:
    """Accept both kwargs-only (RFC 001) and legacy positional path signatures.

    Mirrors ``ChromaBackend._normalize_get_collection_args`` so mempalace's
    ``palace.get_collection(palace_path, collection_name=...)`` flow works
    unchanged. ``create`` and ``options`` are accepted but unused — the
    postgres backend is stateless w.r.t. palace existence.
    """
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
    def __init__(self, pool: ConnectionPool, palace_key: str, collection_name: str):
        self._pool = pool
        self._palace = palace_key
        self._coll = _sanitize_collection(collection_name)

    # ---------------- writes ----------------

    def add(self, *, documents, ids, metadatas=None, embeddings=None):
        self._upsert(documents, ids, metadatas, embeddings, mode="add")

    def upsert(self, *, documents, ids, metadatas=None, embeddings=None):
        self._upsert(documents, ids, metadatas, embeddings, mode="upsert")

    def _upsert(self, documents, ids, metadatas, embeddings, mode):
        n = len(ids)
        if len(documents) != n:
            raise ValueError("documents and ids length mismatch")
        if metadatas is None:
            metadatas = [{}] * n
        elif len(metadatas) != n:
            raise ValueError("metadatas length mismatch")
        if embeddings is None:
            embeddings = _DefaultEmbedder.embed(documents)
        elif len(embeddings) != n:
            raise ValueError("embeddings length mismatch")

        rows = [
            (
                self._palace,
                self._coll,
                ids[i],
                documents[i],
                json.dumps(metadatas[i] or {}),
                _vector_literal(embeddings[i]),
            )
            for i in range(n)
        ]
        if mode == "upsert":
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
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)

    # ---------------- reads ----------------

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
                [r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"]) for r in rows]
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
        out_metas = [
            r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"])
            for r in rows
        ] if spec.metadatas else []
        out_embs = [list(r["embedding"]) for r in rows] if spec.embeddings else None

        return GetResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            embeddings=out_embs,
        )

    def delete(self, *, ids=None, where=None):
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
            f"DELETE FROM mempalace_drawers "
            f"WHERE palace_id = %s AND collection_name = %s AND {' AND '.join(filters)}"
        )
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)

    def count(self) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM mempalace_drawers "
                    "WHERE palace_id = %s AND collection_name = %s",
                    (self._palace, self._coll),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


_SCHEMA_SQL = f"""
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

CREATE INDEX IF NOT EXISTS mempalace_drawers_embedding_idx
    ON mempalace_drawers USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS mempalace_drawers_palace_idx
    ON mempalace_drawers (palace_id, collection_name);

CREATE INDEX IF NOT EXISTS mempalace_drawers_metadata_idx
    ON mempalace_drawers USING GIN (metadata);
"""


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
            configure=lambda conn: register_vector(conn),
            open=True,
        )
        self._schema_ready = False
        self._schema_lock = threading.Lock()
        self._closed = False

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
        return PostgresCollection(self._pool, _palace_key(palace_ref), collection_name)

    def close_palace(self, palace) -> None:
        # Stateless — no per-palace handles to evict.
        return None

    def close(self) -> None:
        if self._closed:
            return
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
