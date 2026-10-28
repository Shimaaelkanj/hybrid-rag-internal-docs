
from __future__ import annotations

import argparse
import shutil
from dataclasses import asdict
from pathlib import Path

from app.chunking import STRATEGIES, build_chunks
from app.config import DEDUP_THRESHOLD, RAW_DIR
from app.embeddings import embed_texts
from app.indexing import (add_document_chunks, get_collection, load_chunk_rows,
                          log_duplicates, rebuild_bm25, reset_indexes,
                          save_chunk_rows, verify_sync)
from app.loaders import LOADERS, load_document, save_processed


def ingest_path(path: Path, strategy: str, reset: bool = False) -> None:
    if reset:
        reset_indexes()

    if path.is_file():
        files = [path]
    else:
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in LOADERS)
    if not files:
        print("No supported files found (.md .txt .html .pdf).")
        return

    collection = get_collection()
    rows = load_chunk_rows()
    total_in, total_dups = 0, 0

    for f in files:
        # keep the raw file so we can re-index later without re-uploading
        raw_copy = RAW_DIR / f.name
        if not (raw_copy.exists() and raw_copy.samefile(f)):
            shutil.copy2(f, raw_copy)

        doc = load_document(raw_copy)
        save_processed(doc)

        chunks = build_chunks(doc, strategy, embed_fn=embed_texts)
        if not chunks:
            print(f"  {f.name}: no text extracted, skipped")
            continue

        vectors = embed_texts([c.text for c in chunks])
        accepted, dups = add_document_chunks(chunks, vectors, collection, DEDUP_THRESHOLD)
        log_duplicates(dups)

        rows = [r for r in rows if r["source"] != doc.source] + [asdict(c) for c in accepted]
        total_in += len(accepted)
        total_dups += len(dups)
        print(f"  {f.name}: {len(accepted)} chunks kept, {len(dups)} duplicates skipped "
              f"[{strategy}]")

    save_chunk_rows(rows)
    rebuild_bm25(rows)

    print(f"\nDone. Added {total_in} chunks, skipped {total_dups} duplicates.")
    print(f"Chunks in store: {len(rows)} | in Chroma: {collection.count()}")
    print("Indexes in sync:", "YES" if verify_sync(collection) else "NO (investigate!)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest documents into the hybrid index")
    ap.add_argument("path", type=Path, help="file or folder to ingest")
    ap.add_argument("--strategy", choices=STRATEGIES, default="recursive")
    ap.add_argument("--reset", action="store_true",
                    help="wipe existing indexes first (use when switching strategy)")
    args = ap.parse_args()
    ingest_path(args.path, args.strategy, args.reset)