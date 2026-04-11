"""
LLM entity/relationship extraction from propositions.

Uses Flash for structured JSON output from clean proposition inputs.
Includes post-extraction validation (stoplist, min length, orphan check).
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from loom.llm.provider import LLMProvider
    from loom.ingestion.propositions import Proposition

ENTITY_STOPLIST = {
    "the model", "this approach", "the method", "our system", "the results",
    "this paper", "the authors", "previous work", "the proposed",
    "the algorithm", "the network", "this work", "the framework",
    "the system", "the data", "the dataset", "the task", "the problem",
    "section", "table", "figure", "equation", "appendix",
}

EXTRACTION_PROMPT = """Extract entities and relationships from these research propositions.

Entity types: concept, technique, paper, claim, metric, system, method, dataset
Relationship types: supports, contradicts, builds_on, compares, component_of, improves, requires, evaluates, related_to

Rules:
- Extract SPECIFIC, CONCRETE entities (paper names, technique names, specific claims with numbers)
- NOT vague terms like "the model", "this approach", "our method"
- Each entity must have a meaningful description
- Each relationship must connect two extracted entities

--- EXAMPLE ---
Propositions:
- "Matcha-TTS uses optimal transport conditional flow matching for speech synthesis."
- "Matcha-TTS achieves a real-time factor (RTF) below 0.1."

Output:
```json
{{
  "entities": [
    {{"name": "Matcha-TTS", "type": "system", "description": "Flow-matching TTS system achieving RTF < 0.1"}},
    {{"name": "optimal transport conditional flow matching", "type": "technique", "description": "Generative modeling via optimal transport paths"}}
  ],
  "relationships": [
    {{"source": "Matcha-TTS", "target": "optimal transport conditional flow matching", "type": "component_of", "description": "Uses OT-CFM as generative backbone"}}
  ]
}}
```

--- YOUR TURN ---
Propositions:
{propositions}

Return ONLY the JSON:"""


def _extract_batch_entities(
    llm: LLMProvider, batch: list[Proposition], doc_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract entities/rels from one batch of propositions."""
    prop_text = "\n".join(f"- \"{p.text}\"" for p in batch)
    prompt = EXTRACTION_PROMPT.format(propositions=prop_text[:5000])
    try:
        resp = llm.generate(prompt, model="flash", temperature=0.1, max_output_tokens=4096)
        parsed = _parse_json(resp.text)
    except Exception:
        parsed = None

    if not parsed:
        return [], []

    entities: list[dict[str, Any]] = []
    for e in parsed.get("entities", []):
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", "")).strip()
        if _validate_entity(name, e.get("description", "")):
            entities.append({
                "name": name,
                "type": e.get("type", "concept"),
                "description": str(e.get("description", "")),
                "source_doc_id": doc_id,
            })

    rels: list[dict[str, Any]] = []
    entity_names = {e["name"].lower() for e in entities}
    for r in parsed.get("relationships", []):
        if not isinstance(r, dict):
            continue
        source = str(r.get("source", "")).strip()
        target = str(r.get("target", "")).strip()
        if source.lower() in entity_names and target.lower() in entity_names:
            rels.append({
                "source_name": source, "target_name": target,
                "type": r.get("type", "related_to"),
                "description": str(r.get("description", "")),
                "source_doc_id": doc_id,
            })

    return entities, rels


def extract_entities_and_relationships(
    llm: LLMProvider,
    propositions: list[Proposition],
    doc_id: str,
    *,
    batch_size: int = 30,
    max_workers: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract entities and relationships from propositions using Flash (parallel batches)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batches = [propositions[i:i + batch_size] for i in range(0, len(propositions), batch_size)]
    batch_results: list[tuple[list, list]] = [None] * len(batches)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_extract_batch_entities, llm, batch, doc_id): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                batch_results[idx] = future.result()
            except Exception:
                batch_results[idx] = ([], [])

    all_entities: list[dict[str, Any]] = []
    all_relationships: list[dict[str, Any]] = []
    seen_entity_names: set[str] = set()

    for entities, rels in batch_results:
        if entities is None:
            continue
        for e in entities:
            if e["name"].lower() not in seen_entity_names:
                seen_entity_names.add(e["name"].lower())
                all_entities.append(e)
        for r in rels:
            src = r["source_name"].lower()
            tgt = r["target_name"].lower()
            if src in seen_entity_names and tgt in seen_entity_names:
                all_relationships.append(r)

    return all_entities, all_relationships


def _validate_entity(name: str, description: str) -> bool:
    if len(name) < 3:
        return False
    if name.lower() in ENTITY_STOPLIST:
        return False
    if not description or description.strip().lower() == name.strip().lower():
        return False
    if re.match(r"^(section|table|figure|equation)\s*\d+", name, re.IGNORECASE):
        return False
    return True


def _parse_json(text: str) -> dict[str, Any] | None:
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                continue
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return None
