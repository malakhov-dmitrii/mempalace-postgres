"""Quick embedder benchmark: chroma-onnx vs sentence-transformers (CPU/MPS/CUDA).

Run:

    python scripts/bench_embed.py
    python scripts/bench_embed.py --n 500

Prints throughput for each embedder the machine can load. Does not touch
Postgres — purely isolates the embedding hot path.
"""

from __future__ import annotations

import argparse
import os
import time


def _sample_docs(n: int) -> list[str]:
    base = [
        "Promo widget extraction status for sprint 32",
        "Тимур Раджабов ведёт продуктовый бэклог команды Promo Tools",
        "pgvector HNSW индекс на embedding vector_cosine_ops",
        "def process_file(filepath: Path, project_path: Path, collection, wing: str)",
        "Daily sync teleprompter — structure for standup talking points",
        "wDrops 2.0 epic PROM-296 deadline sprint 33",
        "MR review в стиле Кузьмича — краткий и по делу",
        "Lobby v3 entry point отсутствует, используется v4 через webpack config",
    ]
    out = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


def _bench(label: str, fn, docs: list[str]) -> float:
    # Warm-up: first call loads the model / JIT compiles.
    fn(docs[:4])
    t0 = time.perf_counter()
    fn(docs)
    elapsed = time.perf_counter() - t0
    rate = len(docs) / elapsed if elapsed else float("inf")
    print(f"  {label:<40} {elapsed:7.2f}s   {rate:8.1f} docs/s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="document count")
    args = parser.parse_args()

    docs = _sample_docs(args.n)
    print(f"Benchmark: embedding {len(docs)} docs (once, warmed up)")
    print("-" * 64)

    # 1. chroma-onnx (CPU)
    try:
        import chromadb.utils.embedding_functions as ef

        onnx = ef.DefaultEmbeddingFunction()
        _bench("chroma-onnx (CPU)", onnx, docs)
    except Exception as e:
        print(f"  chroma-onnx (CPU)                        skipped: {e}")

    # 2. sentence-transformers on available devices
    try:
        from sentence_transformers import SentenceTransformer
        import torch

        devices = ["cpu"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            devices.append("cuda")
        for dev in devices:
            try:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
                _bench(
                    f"sentence-transformers ({dev})",
                    lambda d, _m=model: _m.encode(
                        d, convert_to_numpy=True, show_progress_bar=False
                    ),
                    docs,
                )
            except Exception as e:
                print(f"  sentence-transformers ({dev})           skipped: {e}")
    except ImportError:
        print("  sentence-transformers                     not installed")


if __name__ == "__main__":
    main()
