# mempalace-postgres

Postgres + [pgvector](https://github.com/pgvector/pgvector) backend for [MemPalace](https://github.com/MemPalace/mempalace).

Drop-in replacement for the default embedded ChromaDB backend — plugs in via MemPalace's `mempalace.backends` entry-point contract. One Postgres instance can host every palace across every project, removing the scale ceiling of a single local HNSW index and enabling shared/team-wide memory.

## Why

MemPalace ships with ChromaDB, which is embedded (SQLite + HNSW) and stores everything in a single collection. Past the 40k–200k drawer mark the HNSW index starts corrupting (`Error loading hnsw index`, `Error finding id`, `too many SQL variables`). A few of the open tickets tracking this: [#211](https://github.com/MemPalace/mempalace/issues/211), [#344](https://github.com/MemPalace/mempalace/issues/344), [#444](https://github.com/MemPalace/mempalace/issues/444), [#688](https://github.com/MemPalace/mempalace/issues/688), [#832](https://github.com/MemPalace/mempalace/issues/832).

With this backend:

- One shared table scoped by `(palace_id, collection_name)` — every project gets an isolated logical palace without a separate index file.
- `pgvector` HNSW (`vector_cosine_ops`) handles millions of vectors without the embedded-HNSW failure modes.
- Zero write contention across concurrent MCP servers — one DB process, many thin clients, instead of N Python processes each holding their own HNSW in memory.
- Remote-friendly: point every machine and every Claude session at the same Postgres; shared team memory becomes trivial.

## Requirements

- Postgres 14+ with the `vector` extension available
- Python ≥ 3.11
- MemPalace ≥ 3.3.0

## Install

```bash
pip install mempalace-postgres
```

Until published on PyPI, install directly from GitHub:

```bash
pip install git+https://github.com/malakhov-dmitrii/mempalace-postgres
```

## Configure

```bash
export MEMPALACE_BACKEND=postgres
export MEMPALACE_PG_DSN="postgres://user:password@host:5432/dbname"
# optional
export MEMPALACE_PG_POOL_MAX=10         # default 10
export MEMPALACE_PG_EMBED_DIM=384       # default 384 (matches Chroma's default embedder)
```

That's it. Run `mempalace mine`, `mempalace search`, the MCP server — they all use Postgres.

> **Note:** MemPalace core currently hard-codes its default backend to ChromaDB in `palace.py`, ignoring `MEMPALACE_BACKEND` despite the registry supporting it. Until [PR #1072](https://github.com/MemPalace/mempalace/pull/1072) lands, apply the same patch manually or install MemPalace from the branch.

## Schema

The backend creates its schema lazily on first use:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE mempalace_drawers (
    palace_id TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    document TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (palace_id, collection_name, doc_id)
);

CREATE INDEX mempalace_drawers_embedding_idx
    ON mempalace_drawers USING hnsw (embedding vector_cosine_ops);
CREATE INDEX mempalace_drawers_palace_idx
    ON mempalace_drawers (palace_id, collection_name);
CREATE INDEX mempalace_drawers_metadata_idx
    ON mempalace_drawers USING GIN (metadata);
```

`palace_id` is the value MemPalace passes for the palace — the local palace path for file-rooted palaces.

## Embeddings

Writes that don't supply precomputed embeddings are embedded with Chroma's default `all-MiniLM-L6-v2` (384-dim ONNX bundle) to stay compatible with palaces previously filed through the Chroma backend. Override the dimension with `MEMPALACE_PG_EMBED_DIM` if you swap the embedder.

## Supported features

- `add`, `upsert`, `update`, `query`, `get`, `delete`, `count`
- `where` filters: `$eq`, `$ne`, `$in`, `$nin`, `$and`, `$or`, `$contains`, `$gt`, `$gte`, `$lt`, `$lte`
- `where_document`: `$contains` (case-insensitive `ILIKE`)
- `include` flags: `documents`, `metadatas`, `distances`, `embeddings`

## Limitations / status

- Alpha. Tested against MemPalace 3.3.2.
- Single embedding dimension per deployment — changing `MEMPALACE_PG_EMBED_DIM` after the first write requires dropping the table.
- No migration tool from an existing Chroma palace yet — re-mine projects to populate Postgres.

## License

MIT
