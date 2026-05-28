"""MCP tools backed directly by mempalace-postgres.

The upstream ``mempalace.mcp_server`` still opens a local Chroma palace for
its read tools. This server keeps the same stdio MCP deployment shape while
reading the shared Postgres/pgvector store used by the V1 stack.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Optional

import psycopg
from mcp.server.fastmcp import FastMCP
from psycopg.rows import dict_row

from .backend import PostgresBackend


DEFAULT_PALACE_ID = os.environ.get("MEMPALACE_DEFAULT_PALACE_ID", "/curated")
DEFAULT_COLLECTION = os.environ.get("MEMPALACE_DEFAULT_COLLECTION", "default")

app = FastMCP("mempalace-postgres")
_backend: Optional[PostgresBackend] = None


def _backend_instance() -> PostgresBackend:
    global _backend
    if _backend is None:
        _backend = PostgresBackend()
    return _backend


def _collection():
    return _backend_instance().get_collection(
        DEFAULT_PALACE_ID,
        collection_name=DEFAULT_COLLECTION,
        create=False,
    )


def _dsn() -> str:
    dsn = os.environ.get("MEMPALACE_PG_DSN")
    if not dsn:
        raise RuntimeError("MEMPALACE_PG_DSN is required")
    return dsn


def _jsonable(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _metadata(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return _jsonable(value)
    return _jsonable(json.loads(value))


def _connect():
    return psycopg.connect(_dsn(), row_factory=dict_row)


@app.tool()
def mempalace_status() -> dict:
    """Return total drawer count and wing/room breakdown for the default palace."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT
              COALESCE(NULLIF(metadata->>'wing', ''), 'uncategorized') AS wing,
              COALESCE(NULLIF(metadata->>'room', ''), 'uncategorized') AS room,
              count(*)::int AS drawers
            FROM mempalace_drawers
            WHERE palace_id = %s AND collection_name = %s
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            (DEFAULT_PALACE_ID, DEFAULT_COLLECTION),
        ).fetchall()

    wings: dict[str, dict] = {}
    total = 0
    for row in rows:
        wing = row["wing"]
        room = row["room"]
        drawers = row["drawers"]
        total += drawers
        entry = wings.setdefault(wing, {"drawers": 0, "rooms": {}})
        entry["drawers"] += drawers
        entry["rooms"][room] = drawers

    return {
        "palace_id": DEFAULT_PALACE_ID,
        "collection": DEFAULT_COLLECTION,
        "drawers": total,
        "wing_count": len(wings),
        "room_count": sum(len(v["rooms"]) for v in wings.values()),
        "wings": wings,
    }


@app.tool()
def mempalace_list_wings() -> dict:
    """List wings with drawer counts."""
    status = mempalace_status()
    return {
        "palace_id": status["palace_id"],
        "collection": status["collection"],
        "wings": [
            {"wing": wing, "drawers": data["drawers"], "room_count": len(data["rooms"])}
            for wing, data in status["wings"].items()
        ],
    }


@app.tool()
def mempalace_list_rooms(wing: str) -> dict:
    """List rooms and drawer counts for one wing."""
    status = mempalace_status()
    rooms = status["wings"].get(wing, {}).get("rooms", {})
    return {
        "wing": wing,
        "rooms": [{"room": room, "drawers": drawers} for room, drawers in rooms.items()],
    }


@app.tool()
def mempalace_get_taxonomy() -> dict:
    """Return the complete wing to room taxonomy."""
    return mempalace_status()


@app.tool()
def mempalace_list_drawers(
    wing: Optional[str] = None,
    room: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """List drawers with optional wing/room filters and pagination."""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    filters = ["palace_id = %s", "collection_name = %s"]
    params: list = [DEFAULT_PALACE_ID, DEFAULT_COLLECTION]
    if wing:
        filters.append("metadata->>'wing' = %s")
        params.append(wing)
    if room:
        filters.append("metadata->>'room' = %s")
        params.append(room)
    where = " AND ".join(filters)

    with _connect() as conn:
        total = conn.execute(
            f"SELECT count(*)::int AS total FROM mempalace_drawers WHERE {where}",
            params,
        ).fetchone()["total"]
        rows = conn.execute(
            f"""
            SELECT doc_id, document, metadata, created_at
            FROM mempalace_drawers
            WHERE {where}
            ORDER BY created_at DESC, doc_id
            LIMIT %s OFFSET %s
            """,
            [*params, limit, offset],
        ).fetchall()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "drawers": [
            {
                "id": row["doc_id"],
                "preview": row["document"][:300],
                "metadata": _metadata(row["metadata"]),
                "created_at": _jsonable(row["created_at"]),
            }
            for row in rows
        ],
    }


@app.tool()
def mempalace_get_drawer(drawer_id: str) -> dict:
    """Fetch one drawer by ID."""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT doc_id, document, metadata, created_at
            FROM mempalace_drawers
            WHERE palace_id = %s AND collection_name = %s AND doc_id = %s
            """,
            (DEFAULT_PALACE_ID, DEFAULT_COLLECTION, drawer_id),
        ).fetchone()
    if row is None:
        return {"found": False, "id": drawer_id}
    return {
        "found": True,
        "id": row["doc_id"],
        "document": row["document"],
        "metadata": _metadata(row["metadata"]),
        "created_at": _jsonable(row["created_at"]),
    }


@app.tool()
def mempalace_search(
    query: str,
    limit: int = 5,
    wing: Optional[str] = None,
    room: Optional[str] = None,
    max_distance: float = 1.5,
    context: Optional[str] = None,
) -> dict:
    """Semantic search against the default Postgres/pgvector palace."""
    del context
    limit = max(1, min(int(limit), 100))
    where = {}
    if wing:
        where["wing"] = wing
    if room:
        where["room"] = room

    kwargs = {
        "query_texts": [query],
        "n_results": limit,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    result = _collection().query(**kwargs)

    ids = result.ids[0] if result.ids else []
    documents = result.documents[0] if result.documents else []
    metadatas = result.metadatas[0] if result.metadatas else []
    distances = result.distances[0] if result.distances else []

    results = []
    for idx, drawer_id in enumerate(ids):
        distance = distances[idx] if idx < len(distances) else None
        if max_distance and distance is not None and distance > max_distance:
            continue
        results.append(
            {
                "id": drawer_id,
                "document": documents[idx] if idx < len(documents) else "",
                "metadata": _jsonable(metadatas[idx]) if idx < len(metadatas) else {},
                "distance": distance,
            }
        )

    return {
        "query": query,
        "palace_id": DEFAULT_PALACE_ID,
        "collection": DEFAULT_COLLECTION,
        "count": len(results),
        "results": results,
    }


@app.tool()
def mempalace_check_duplicate(content: str, limit: int = 5, max_distance: float = 0.15) -> dict:
    """Find near-duplicate drawers for proposed content."""
    result = mempalace_search(content[:250], limit=limit, max_distance=max_distance)
    return {"duplicate_count": result["count"], "matches": result["results"]}


@app.tool()
def mempalace_reconnect() -> dict:
    """Reset the backend connection pool on the next tool call."""
    global _backend
    if _backend is not None:
        _backend.close()
        _backend = None
    return {"ok": True, "message": "Postgres backend will reconnect on next call"}


def main() -> None:
    app.run("stdio")


if __name__ == "__main__":
    main()
