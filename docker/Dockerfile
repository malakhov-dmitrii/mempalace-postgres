FROM python:3.12-slim

ENV MEMPALACE_BACKEND=postgres
ENV MEMPALACE_PG_EMBEDDER=chroma-onnx

WORKDIR /opt/mempalace-postgres
COPY pyproject.toml README.md LICENSE ./
COPY mempalace_postgres ./mempalace_postgres

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
    "mcp-proxy==0.12.0" \
    "mempalace>=3.3.5" \
    "psycopg[binary,pool]>=3.2" \
    "pgvector>=0.3" \
    .

EXPOSE 8811

ENTRYPOINT ["mcp-proxy"]
