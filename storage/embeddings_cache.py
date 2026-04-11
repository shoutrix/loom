"""
Local FAISS index persistence and embedding cache.

Saves and loads FAISS indexes + metadata to/from disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from loom.search.semantic import DualSemanticIndex, FAISSIndex

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def save_index(index: FAISSIndex, path: Path) -> None:
    """Save a FAISS index and its metadata to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "ids": index.ids,
        "texts": index.texts,
        "doc_ids": index.doc_ids,
        "dimension": index.dimension,
    }
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    if HAS_FAISS:
        faiss.write_index(index._index, str(path.with_suffix(".faiss")))
    else:
        vecs = np.stack(index._vectors) if index._vectors else np.empty((0, index.dimension))
        np.save(str(path.with_suffix(".npy")), vecs)


def load_index(index: FAISSIndex, path: Path) -> bool:
    """Load a FAISS index and metadata from disk. Returns True on success."""
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return False

    with open(meta_path) as f:
        meta = json.load(f)

    index.ids = meta["ids"]
    index.texts = meta["texts"]
    index.doc_ids = meta["doc_ids"]

    if HAS_FAISS:
        faiss_path = path.with_suffix(".faiss")
        if faiss_path.exists():
            index._index = faiss.read_index(str(faiss_path))
            return True
    else:
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            vecs = np.load(str(npy_path))
            index._vectors = [vecs[i] for i in range(vecs.shape[0])]
            return True

    return False


def save_dual_index(dual_index: DualSemanticIndex, data_dir: Path) -> None:
    """Save both chunk and proposition indexes."""
    from loom.search.semantic import FAISSIndex
    save_index(dual_index.chunk_index, data_dir / "chunks_index")
    save_index(dual_index.proposition_index, data_dir / "props_index")


def load_dual_index(dual_index: DualSemanticIndex, data_dir: Path) -> bool:
    """Load both indexes from disk. Returns True if at least one loaded."""
    from loom.search.semantic import FAISSIndex
    a = load_index(dual_index.chunk_index, data_dir / "chunks_index")
    b = load_index(dual_index.proposition_index, data_dir / "props_index")
    return a or b
