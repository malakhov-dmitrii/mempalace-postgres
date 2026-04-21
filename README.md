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

# optional — pool + schema
export MEMPALACE_PG_POOL_MAX=10         # default 10
export MEMPALACE_PG_EMBED_DIM=384       # default 384
export MEMPALACE_PG_SKIP_HNSW=1         # create HNSW index manually after bulk import

# optional — write batching (huge speed win during mine)
export MEMPALACE_PG_BATCH=256           # flush buffer at N drawers (default 256)
export MEMPALACE_PG_BATCH_TTL=5         # or every N seconds (default 5)

# optional — embedder selection
export MEMPALACE_PG_EMBEDDER=auto                              # auto | sentence-transformers | chroma-onnx
export MEMPALACE_PG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export MEMPALACE_PG_EMBED_DEVICE=cpu                           # auto | cpu | mps | cuda
```

Run `mempalace mine`, `mempalace search`, the MCP server — they all use Postgres.

## Performance

Two design choices inherited from MemPalace core make the default miner slow: `add_drawer` calls `collection.upsert` once per chunk, and the embedding function it reaches through is Chroma's bundled ONNX CPU model.

This backend fixes both without any upstream change:

1. **Buffered writes.** `add` / `upsert` enqueue into an in-memory buffer per collection and flush in bulk — one batched embedding call + one `executemany` — when the buffer crosses `MEMPALACE_PG_BATCH` entries, when `MEMPALACE_PG_BATCH_TTL` seconds elapse, or when a read / delete / `close()` / process exit forces a flush. Reads always see a consistent view because they flush first.
2. **Faster embedder when available.** With `sentence-transformers` installed, the backend uses it automatically — batch encoding through PyTorch is ~12× faster than Chroma's per-call ONNX at the same model (`all-MiniLM-L6-v2`), on CPU. For bigger models the same knob exposes CUDA. MPS is skipped by `auto` on purpose (see below).

Micro-bench on Apple M-series, `scripts/bench_embed.py --n 300`:

| Embedder                              | docs/s | speedup |
|---------------------------------------|-------:|--------:|
| `chroma-onnx` (CPU, baseline)         |   ~73  |     1× |
| `sentence-transformers` (CPU / torch) |  ~919  |   12.6× |
| `sentence-transformers` (MPS)         |  ~244  |    3.4× |

MPS is slower than CPU here because all-MiniLM-L6-v2 is tiny (22M params) and the kernel-launch overhead dominates. For heavier embedders MPS can still win — request it explicitly with `MEMPALACE_PG_EMBED_DEVICE=mps`.

### Bulk imports

For very large initial imports (tens of thousands of drawers) skip the HNSW index during the load and build it once at the end:

```bash
# first load
export MEMPALACE_PG_SKIP_HNSW=1
mempalace mine /path/to/project
# ...more mines...

# then build the index in one pass
psql ... -c "CREATE INDEX mempalace_drawers_embedding_idx \
             ON mempalace_drawers USING hnsw (embedding vector_cosine_ops);"
unset MEMPALACE_PG_SKIP_HNSW
```

This is dramatically faster than updating the graph on every insert and avoids the index-bloat pattern seen in [mempalace#344](https://github.com/MemPalace/mempalace/issues/344).

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
