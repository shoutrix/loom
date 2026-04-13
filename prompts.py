"""
Centralized LLM prompt templates for Loom.

All prompts used across the system live here for easy tuning,
consistency, and visibility into what the LLM is being asked.
"""

# ── Chat ─────────────────────────────────────────────────────────────

CHAT_SYSTEM = (
    "You are Loom, a research knowledge assistant. You answer questions "
    "about the user's research library using retrieved context from their "
    "documents and knowledge graph.\n\n"
    "Rules:\n"
    "- Ground every claim in the retrieved context\n"
    "- Cite specific papers/documents when possible using [doc_id] notation\n"
    "- If the context doesn't contain enough information, say so clearly\n"
    "- Be precise and technical\n"
    "- Highlight cross-domain connections when relevant\n"
    "- Structure longer answers with clear headings"
)

CHAT_ANSWER = (
    "Answer the question using the retrieved context. "
    "Cite sources using [doc_id]. If the context is insufficient, say so."
)


# ── Paper Search: Planner ────────────────────────────────────────────

SEARCH_PLAN = (
    "You are an expert research librarian. Given a user's research query, "
    "generate 4-5 distinct search angles to find the most relevant academic papers.\n\n"
    "For EACH angle, produce a tailored query string for each API:\n"
    "- semantic_scholar: natural language sentence (S2 uses embedding-based search, so full sentences work best)\n"
    "- arxiv: short 2-5 word keyword phrase (arXiv search is keyword-based, keep it tight)\n"
    "- openalex: keyword query with the most important terms\n\n"
    "The angles should cover:\n"
    "1. The direct/literal interpretation of the query\n"
    "2. The core technical problem or method\n"
    "3. Closely related subfields or alternative framings\n"
    "4. Foundational/seminal work in this area\n"
    "5. (Optional) Recent surveys or benchmarks\n\n"
    "Make each query genuinely different — avoid rephrasing the same thing.\n\n"
    "User Query: {query}\n\n"
    "JSON schema:\n{schema}"
)


# ── Paper Search: LLM Relevance Filter ───────────────────────────────

RELEVANCE_SCORE = (
    "You are a research assistant evaluating papers for relevance to a query.\n\n"
    "QUERY: {query}\n\n"
    "PAPERS:\n{papers_json}\n\n"
    "For EACH paper, assign a relevance score from 0-10:\n"
    "  0-4: Not relevant (different topic, tangentially related at best)\n"
    "  5-6: Somewhat relevant (related background, useful context)\n"
    "  7-8: Relevant (directly addresses the topic or a key sub-problem)\n"
    "  9-10: Highly relevant (core paper for this query)\n\n"
    "When in doubt, lean towards including (score 5+) rather than excluding.\n"
    "Papers that provide important foundational context should score 5-6.\n\n"
    "Return a JSON array:\n"
    '[{{"id": "...", "score": 7, "rationale": "one-sentence reason"}}]\n'
    "Return ONLY the JSON array, no other text."
)


# ── Paper Search: Root Paper Discovery ───────────────────────────────

ROOT_PAPER_JUDGE = (
    "You are identifying the foundational papers that originated a research direction.\n\n"
    "Research query: {query}\n\n"
    "These papers were discovered by tracing backwards through the citation graph "
    "of the top search results. They are cited (directly or transitively) by many "
    "of the relevant papers.\n\n"
    "Candidate root papers:\n{candidates}\n\n"
    "For each candidate, decide:\n"
    "- Is this paper genuinely foundational for the research direction described "
    "by the query? (not just a generally popular paper)\n"
    "- Score its 'foundational importance' from 0-10\n"
    "- One sentence explaining why it's foundational (or why it isn't)\n\n"
    "Return a JSON array:\n"
    '[{{"id": "...", "score": 8, "rationale": "Introduced the core technique..."}}]\n'
    "Return ONLY the JSON array, no other text."
)


# ── Ingestion: Proposition Extraction ────────────────────────────────

PROPOSITION_EXTRACT = (
    "Decompose the following text into atomic, self-contained propositions.\n\n"
    "Rules:\n"
    "- Each proposition should express exactly ONE fact, claim, or relationship\n"
    "- Each proposition must be understandable WITHOUT reading the original text\n"
    "- De-contextualize: replace pronouns and references with their full names\n"
    "- Include specific numbers, metrics, and comparisons\n"
    "- Preserve technical terminology exactly\n"
    "- Skip meta-commentary (\"In this section...\", \"We describe...\")\n"
    "- Output as a JSON array of strings\n\n"
    "Example input:\n"
    "\"Matcha-TTS uses optimal transport conditional flow matching for synthesis. "
    "Unlike diffusion models that require hundreds of steps, it achieves RTF < 0.1 "
    "with only 10 ODE steps.\"\n\n"
    "Example output:\n"
    "[\"Matcha-TTS uses optimal transport conditional flow matching for speech synthesis.\", "
    "\"Diffusion models typically require hundreds of iterative steps for generation.\", "
    "\"Matcha-TTS achieves a real-time factor (RTF) below 0.1.\", "
    "\"Matcha-TTS requires only 10 ODE solver steps for generation.\", "
    "\"Matcha-TTS is significantly faster than diffusion-based speech synthesis models.\"]\n\n"
    "Now decompose this text:\n"
    "---\n{text}\n---\n\n"
    "Output ONLY the JSON array:"
)


# ── Ingestion: Contextual Chunk Enrichment ───────────────────────────

CHUNK_ENRICHMENT = (
    "You are helping prepare document chunks for a semantic search index.\n\n"
    "Given a document's title and abstract/summary, write exactly 2 sentences "
    "that establish the context for any chunk from this document. These sentences "
    "will be prepended to each chunk before embedding.\n\n"
    "Requirements:\n"
    "- Mention the document title or key topic\n"
    "- Establish the domain and main contribution\n"
    "- Be factual and specific, not generic\n"
    "- Keep it under 60 words total\n\n"
    "Document title: {title}\n\n"
    "Abstract/Summary:\n{abstract}\n\n"
    "Write your 2-sentence context prefix:"
)


# ── Knowledge Graph: Entity/Relationship Extraction ──────────────────

ENTITY_EXTRACTION = (
    "Extract entities and relationships from these research propositions.\n\n"
    "Entity types: concept, technique, paper, claim, metric, system, method, dataset\n"
    "Relationship types: supports, contradicts, builds_on, compares, component_of, "
    "improves, requires, evaluates, related_to\n\n"
    "Rules:\n"
    "- Extract SPECIFIC, CONCRETE entities (paper names, technique names, specific claims with numbers)\n"
    "- NOT vague terms like \"the model\", \"this approach\", \"our method\"\n"
    "- Each entity must have a meaningful description\n"
    "- Each relationship must connect two extracted entities\n\n"
    "--- EXAMPLE ---\n"
    "Propositions:\n"
    "- \"Matcha-TTS uses optimal transport conditional flow matching for speech synthesis.\"\n"
    "- \"Matcha-TTS achieves a real-time factor (RTF) below 0.1.\"\n\n"
    "Output:\n"
    "```json\n"
    "{{\n"
    "  \"entities\": [\n"
    "    {{\"name\": \"Matcha-TTS\", \"type\": \"system\", \"description\": \"Flow-matching TTS system achieving RTF < 0.1\"}},\n"
    "    {{\"name\": \"optimal transport conditional flow matching\", \"type\": \"technique\", \"description\": \"Generative modeling via optimal transport paths\"}}\n"
    "  ],\n"
    "  \"relationships\": [\n"
    "    {{\"source\": \"Matcha-TTS\", \"target\": \"optimal transport conditional flow matching\", \"type\": \"component_of\", \"description\": \"Uses OT-CFM as generative backbone\"}}\n"
    "  ]\n"
    "}}\n"
    "```\n\n"
    "--- YOUR TURN ---\n"
    "Propositions:\n{propositions}\n\n"
    "Return ONLY the JSON:"
)


# ── Knowledge Graph: Entity Resolution (Flash batch) ─────────────────

ENTITY_RESOLUTION_FLASH = (
    "For each pair, decide if the new entity should MERGE with the candidate "
    "or be kept as a SEPARATE entity.\n\n"
    "Pairs:\n{pairs}\n\n"
    "Reply with a JSON array of decisions, one per pair: [\"merge\", \"separate\", ...]"
)


# ── Knowledge Graph: Entity Resolution (structured) ──────────────────

ENTITY_RESOLUTION_STRUCTURED = (
    "You are resolving entity references in a knowledge graph.\n"
    "For each entity, decide:\n"
    "- MERGE with candidate [id] if they refer to the same concept\n"
    "- NEW if this is a genuinely distinct concept\n\n"
    "Consider context carefully. Same name ≠ same concept "
    "(e.g., 'transformer' in ML vs electrical engineering).\n\n"
    "{entries}\n\n"
    "Return a JSON array with one entry per entity: "
    "[{{\"action\": \"merge\", \"target_id\": \"xxx\"}}, {{\"action\": \"new\"}}, ...]"
)


# ── Knowledge Graph: Community Summarization ─────────────────────────

COMMUNITY_SUMMARY = (
    "Summarize this cluster of related research concepts.\n\n"
    "Entities:\n{entities}\n\n"
    "Relationships:\n{relationships}\n\n"
    "Write a 3-5 sentence summary covering:\n"
    "1. What this cluster is about (main theme)\n"
    "2. Key insights and contributions\n"
    "3. Any tensions, open questions, or competing approaches\n"
    "Be specific and technical."
)


# ── Knowledge Graph: Cross-Domain Connection Discovery ───────────────

CROSS_DOMAIN_CONNECTION = (
    "You are analyzing potential cross-domain connections in a research knowledge graph.\n\n"
    "For each pair of entities from different research domains, determine:\n"
    "1. Is there a meaningful intellectual connection? (not just superficial keyword overlap)\n"
    "2. If yes, explain the deep connection in 2-3 sentences\n"
    "3. Could understanding from one domain inform the other?\n\n"
    "Pairs:\n{pairs}\n\n"
    "Return a JSON array with one entry per pair:\n"
    '[{{"connected": true, "description": "..."}}, {{"connected": false, "description": ""}}, ...]'
)
