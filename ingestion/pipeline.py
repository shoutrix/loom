"""
Master ingestion pipeline.

Orchestrates: parse -> chunk -> enrich -> propositions -> embed -> extract -> resolve -> ingest

Uses ThreadPoolExecutor for parallelism across independent LLM/embedding batches.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.config import Settings
    from loom.llm.provider import LLMProvider
    from loom.llm.embeddings import EmbeddingProvider
    from loom.graph.store import GraphStore
    from loom.search.semantic import DualSemanticIndex
    from loom.search.keyword import KeywordIndex
    from loom.storage.vault import VaultManager


@dataclass
class IngestionResult:
    doc_id: str
    title: str
    num_chunks: int = 0
    num_propositions: int = 0
    num_entities: int = 0
    num_relationships: int = 0
    num_communities_dirtied: int = 0
    llm_calls_flash: int = 0
    llm_calls_pro: int = 0
    elapsed_seconds: float = 0.0
    step_timings: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _log(doc_id: str, msg: str) -> None:
    print(f"  [Ingest] [{doc_id}] {msg}", flush=True)


def _timed(doc_id: str, step_name: str, timings: dict):
    """Context manager that times a step and logs it."""
    class _Timer:
        def __enter__(self):
            self.t0 = time.time()
            return self
        def __exit__(self, *_):
            elapsed = time.time() - self.t0
            timings[step_name] = round(elapsed, 2)
            _log(doc_id, f"  {step_name}: {elapsed:.1f}s")
    return _Timer()


class IngestionPipeline:
    """Orchestrates the full document ingestion flow."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        graph_store: GraphStore,
        semantic_index: DualSemanticIndex,
        keyword_index: KeywordIndex,
        vault: VaultManager,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.embedder = embedder
        self.graph = graph_store
        self.semantic_index = semantic_index
        self.keyword_index = keyword_index
        self.vault = vault
        self._docs_since_deep_scan = 0

    def ingest_document(self, parsed_doc) -> IngestionResult:
        """Run full ingestion pipeline on a parsed document."""
        from loom.ingestion.chunker import chunk_document
        from loom.ingestion.enrichment import generate_context_prefix, enrich_chunks
        from loom.ingestion.propositions import extract_propositions
        from loom.graph.extractor import extract_entities_and_relationships
        from loom.graph.resolver import resolve_entities
        from loom.graph.hierarchy import update_communities

        t0 = time.time()
        calls_before = self.llm.usage.total_calls
        result = IngestionResult(doc_id=parsed_doc.doc_id, title=parsed_doc.title)
        doc_id = parsed_doc.doc_id
        timings = result.step_timings

        try:
            # --- Step 1: Chunk ---
            with _timed(doc_id, "chunk", timings):
                chunks = chunk_document(doc_id, parsed_doc.content)
                result.num_chunks = len(chunks)

            if not chunks:
                result.errors.append("No chunks produced from document")
                return result
            _log(doc_id, f"  -> {len(chunks)} chunks")

            # --- Step 2: Context enrichment + Propositions (parallel) ---
            # These are independent: enrichment uses abstract, propositions use chunks
            with _timed(doc_id, "enrich+propositions", timings):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    enrich_future = pool.submit(
                        generate_context_prefix,
                        self.llm,
                        parsed_doc.title,
                        parsed_doc.abstract or parsed_doc.content[:500],
                    )
                    props_future = pool.submit(
                        extract_propositions, self.llm, chunks, doc_id,
                    )
                    context_prefix = enrich_future.result()
                    propositions = props_future.result()

                enriched_texts = enrich_chunks(chunks, context_prefix)
                result.num_propositions = len(propositions)

            _log(doc_id, f"  -> {len(propositions)} propositions")

            # --- Step 3: Embedding (chunks + propositions in parallel) ---
            with _timed(doc_id, "embedding", timings):
                prop_texts = [p.text for p in propositions]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    chunk_emb_future = pool.submit(self.embedder.embed, enriched_texts)
                    if prop_texts:
                        prop_emb_future = pool.submit(self.embedder.embed, prop_texts)
                    else:
                        prop_emb_future = None

                    chunk_embeddings = chunk_emb_future.result()
                    prop_embeddings = prop_emb_future.result() if prop_emb_future else None

            # --- Step 4: Index (fast, in-memory) ---
            with _timed(doc_id, "indexing", timings):
                chunk_ids = [c.chunk_id for c in chunks]
                chunk_texts_raw = [c.text for c in chunks]
                self.semantic_index.add_chunks(
                    chunk_ids, chunk_embeddings, chunk_texts_raw,
                    doc_id=doc_id,
                )
                if propositions and prop_embeddings is not None:
                    prop_ids = [p.prop_id for p in propositions]
                    self.semantic_index.add_propositions(
                        prop_ids, prop_embeddings, prop_texts,
                        doc_id=doc_id,
                    )
                self.keyword_index.add_documents(
                    chunk_ids, chunk_texts_raw, doc_id=doc_id,
                )
                if propositions:
                    self.keyword_index.add_documents(
                        [p.prop_id for p in propositions],
                        prop_texts, doc_id=doc_id, is_proposition=True,
                    )

            # --- Step 5: Entity + relationship extraction ---
            with _timed(doc_id, "entity_extraction", timings):
                raw_entities, raw_relationships = extract_entities_and_relationships(
                    self.llm, propositions, doc_id,
                )
            _log(doc_id, f"  -> {len(raw_entities)} entities, {len(raw_relationships)} rels")

            # --- Step 6: Embed entities ---
            with _timed(doc_id, "entity_embedding", timings):
                if raw_entities:
                    entity_texts = [
                        f"{e['name']}: {e.get('description', '')}" for e in raw_entities
                    ]
                    entity_embeddings = self.embedder.embed(entity_texts)
                else:
                    entity_embeddings = None

            # --- Step 7: Entity resolution ---
            with _timed(doc_id, "entity_resolution", timings):
                uuid_map, new_entity_ids = resolve_entities(
                    self.llm, self.embedder, self.graph,
                    raw_entities, entity_embeddings, doc_id,
                    self.settings.graph,
                )
            _log(doc_id, f"  -> {len(new_entity_ids)} new entities in graph")

            # --- Step 8: Ingest relationships ---
            with _timed(doc_id, "relationships", timings):
                for rel in raw_relationships:
                    src_id = uuid_map.get(rel["source_name"], rel.get("source_id"))
                    tgt_id = uuid_map.get(rel["target_name"], rel.get("target_id"))
                    if src_id and tgt_id and src_id != tgt_id:
                        self.graph.add_relationship(
                            source_id=src_id, target_id=tgt_id,
                            relation_type=rel.get("type", "related_to"),
                            description=rel.get("description", ""),
                            source_doc_id=doc_id,
                        )
            result.num_entities = len(uuid_map)
            result.num_relationships = len(raw_relationships)

            # --- Step 9: Community update ---
            with _timed(doc_id, "communities", timings):
                dirty_count = update_communities(self.llm, self.graph)
                result.num_communities_dirtied = dirty_count

            # --- Step 10: Cross-domain connections ---
            with _timed(doc_id, "connections", timings):
                from loom.graph.connections import detect_bridges, detect_latent_connections
                detect_bridges(self.graph, new_entity_ids)
                detect_latent_connections(self.graph, self.embedder, new_entity_ids)

                self._docs_since_deep_scan += 1
                if self._docs_since_deep_scan >= self.settings.graph.deep_scan_interval:
                    from loom.graph.connections import deep_connection_scan
                    deep_connection_scan(self.llm, self.graph)
                    self._docs_since_deep_scan = 0

            # --- Step 11: Save to vault ---
            with _timed(doc_id, "vault_save", timings):
                self.vault.save_ingested_document(
                    doc_id=doc_id, title=parsed_doc.title,
                    markdown_content=parsed_doc.content,
                    source_type=parsed_doc.source_type,
                    source_url=parsed_doc.source_url,
                )

            total = time.time() - t0
            _log(doc_id, f"DONE in {total:.1f}s | {result.num_chunks} chunks, "
                 f"{result.num_propositions} props, {result.num_entities} entities")

        except Exception as e:
            import traceback
            _log(doc_id, f"ERROR: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            result.errors.append(str(e))

        result.elapsed_seconds = time.time() - t0
        calls_after = self.llm.usage.total_calls
        total_new_calls = calls_after - calls_before
        result.llm_calls_flash = total_new_calls // 2
        result.llm_calls_pro = total_new_calls - result.llm_calls_flash

        return result
