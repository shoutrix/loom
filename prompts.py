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
    "generate 5-6 distinct search angles to find the most relevant academic papers.\n\n"
    "For EACH angle, produce a tailored query string for each API:\n"
    "- semantic_scholar: natural language sentence (S2 uses embedding-based search, so full sentences work best)\n"
    "- arxiv: short 2-5 word keyword phrase (arXiv search is keyword-based, keep it tight)\n"
    "- openalex: keyword query with the most important terms\n\n"
    "The angles MUST cover:\n"
    "1. The direct/literal interpretation of the query\n"
    "2. The core technical problem or method\n"
    "3. Closely related subfields or alternative framings\n"
    "4. Foundational/seminal work in this area\n"
    "5. Component decomposition: search for each major concept in the query independently "
    "(e.g., for 'FSQ-based TTS', search separately for 'finite scalar quantization' "
    "and 'neural codec speech synthesis')\n"
    "6. (Optional) Recent surveys or benchmarks\n\n"
    "Important guidelines:\n"
    "- If you are uncertain about the exact terminology used in this field, "
    "generate BROADER queries that cover adjacent concepts.\n"
    "- Always include both full forms and common acronyms/abbreviations "
    "(e.g., FSQ, TTS, VQ-VAE, ASR, NLP).\n"
    "- Make each query genuinely different — avoid rephrasing the same thing.\n\n"
    "--- EXAMPLE ---\n"
    "User Query: \"Graph neural networks for drug discovery\"\n"
    "Output:\n"
    "```json\n"
    "{{\"queries\": [\n"
    "  {{\"label\": \"direct\", \"semantic_scholar\": \"Graph neural networks applied to molecular property prediction and drug discovery\", \"arxiv\": \"graph neural network drug discovery\", \"openalex\": \"graph neural network molecular drug\"}},\n"
    "  {{\"label\": \"core method\", \"semantic_scholar\": \"Message passing neural networks for predicting molecular interactions and binding affinity\", \"arxiv\": \"message passing molecular graphs\", \"openalex\": \"message passing neural network molecular\"}},\n"
    "  {{\"label\": \"adjacent framing\", \"semantic_scholar\": \"Geometric deep learning on 3D molecular structures for virtual screening\", \"arxiv\": \"geometric deep learning molecules\", \"openalex\": \"geometric deep learning virtual screening\"}},\n"
    "  {{\"label\": \"foundational\", \"semantic_scholar\": \"Convolutional networks on graphs and spectral approaches for learning molecular fingerprints\", \"arxiv\": \"graph convolution molecular fingerprint\", \"openalex\": \"graph convolution molecular representation\"}},\n"
    "  {{\"label\": \"component: GNN architectures\", \"semantic_scholar\": \"Graph attention networks and graph transformer architectures for node and graph classification\", \"arxiv\": \"graph attention transformer\", \"openalex\": \"graph attention network transformer\"}},\n"
    "  {{\"label\": \"surveys\", \"semantic_scholar\": \"Survey of deep learning methods for molecular generation and optimization in drug design\", \"arxiv\": \"survey deep learning drug design\", \"openalex\": \"survey deep learning drug design molecular\"}}\n"
    "]}}\n"
    "```\n\n"
    "--- EXAMPLE ---\n"
    "User Query: \"Smart turn detection for conversational voice agents\"\n"
    "Output:\n"
    "```json\n"
    "{{\"queries\": [\n"
    "  {{\"label\": \"direct\", \"semantic_scholar\": \"Detecting turn-taking cues and endpoint detection in spoken conversational AI agents\", \"arxiv\": \"turn detection voice agent\", \"openalex\": \"turn detection conversational voice agent\"}},\n"
    "  {{\"label\": \"core method\", \"semantic_scholar\": \"End-of-turn prediction and voice activity detection using prosodic and linguistic features\", \"arxiv\": \"end-of-turn prediction prosody\", \"openalex\": \"endpointing voice activity detection\"}},\n"
    "  {{\"label\": \"adjacent\", \"semantic_scholar\": \"Duplex conversation systems and full-duplex speech models for real-time spoken dialogue\", \"arxiv\": \"duplex spoken dialogue system\", \"openalex\": \"full-duplex speech dialogue\"}},\n"
    "  {{\"label\": \"foundational\", \"semantic_scholar\": \"Silence detection and inter-pausal unit segmentation in conversational speech\", \"arxiv\": \"silence detection speech segmentation\", \"openalex\": \"inter-pausal unit conversational speech\"}},\n"
    "  {{\"label\": \"component: VAD models\", \"semantic_scholar\": \"Neural voice activity detection models for streaming audio and real-time speech recognition\", \"arxiv\": \"neural voice activity detection\", \"openalex\": \"neural VAD streaming ASR\"}}\n"
    "]}}\n"
    "```\n\n"
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
    "--- EXAMPLE ---\n"
    "QUERY: \"Efficient inference for large language models\"\n"
    "PAPERS: [{{\"id\": \"s2:a1\", \"title\": \"FlashAttention: Fast and Memory-Efficient Exact Attention\", "
    "\"abstract\": \"We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling...\"}},\n"
    "{{\"id\": \"s2:b2\", \"title\": \"A Survey on RGB-D Salient Object Detection\", "
    "\"abstract\": \"We review methods for detecting salient objects in RGB-D images...\"}},\n"
    "{{\"id\": \"s2:c3\", \"title\": \"Scaling Laws for Neural Language Models\", "
    "\"abstract\": \"We study empirical scaling laws for language model performance...\"}}]\n"
    "Output:\n"
    "[{{\"id\": \"s2:a1\", \"score\": 9, \"rationale\": \"Directly addresses efficient attention computation for LLMs\"}},\n"
    " {{\"id\": \"s2:b2\", \"score\": 1, \"rationale\": \"Computer vision paper, unrelated to LLM inference\"}},\n"
    " {{\"id\": \"s2:c3\", \"score\": 6, \"rationale\": \"Foundational context on LLM scaling but not about inference efficiency\"}}]\n\n"
    "--- EXAMPLE ---\n"
    "QUERY: \"Reinforcement learning for robotic manipulation\"\n"
    "PAPERS: [{{\"id\": \"s2:d4\", \"title\": \"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization\", "
    "\"abstract\": \"We present a method for transferring policies trained in simulation to real robots...\"}},\n"
    "{{\"id\": \"s2:e5\", \"title\": \"BERT: Pre-training of Deep Bidirectional Transformers\", "
    "\"abstract\": \"We introduce BERT, a new language representation model...\"}}]\n"
    "Output:\n"
    "[{{\"id\": \"s2:d4\", \"score\": 8, \"rationale\": \"Directly tackles RL policy transfer for robotic manipulation tasks\"}},\n"
    " {{\"id\": \"s2:e5\", \"score\": 0, \"rationale\": \"NLP model, not related to robotics or reinforcement learning\"}}]\n\n"
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
    "--- EXAMPLE ---\n"
    "Research query: \"Diffusion models for image generation\"\n"
    "Candidates:\n"
    "1. \"Denoising Diffusion Probabilistic Models\" (Ho et al., 2020) — convergence=0.82\n"
    "2. \"Attention Is All You Need\" (Vaswani et al., 2017) — convergence=0.45\n"
    "3. \"Deep Unsupervised Learning using Nonequilibrium Thermodynamics\" (Sohl-Dickstein et al., 2015) — convergence=0.71\n"
    "Output:\n"
    "[{{\"id\": \"s2:x1\", \"score\": 10, \"rationale\": \"Introduced the modern DDPM framework that all subsequent diffusion image generation methods build on\"}},\n"
    " {{\"id\": \"s2:x2\", \"score\": 3, \"rationale\": \"Foundational for deep learning broadly but not specific to diffusion models for images\"}},\n"
    " {{\"id\": \"s2:x3\", \"score\": 9, \"rationale\": \"Originated the thermodynamic diffusion framework that DDPMs later made practical\"}}]\n\n"
    "--- EXAMPLE ---\n"
    "Research query: \"Federated learning for healthcare\"\n"
    "Candidates:\n"
    "1. \"Communication-Efficient Learning of Deep Networks from Decentralized Data\" (McMahan et al., 2017) — convergence=0.90\n"
    "2. \"ImageNet Large Scale Visual Recognition Challenge\" (Russakovsky et al., 2015) — convergence=0.30\n"
    "Output:\n"
    "[{{\"id\": \"s2:y1\", \"score\": 10, \"rationale\": \"Introduced FedAvg, the foundational algorithm for federated learning used across all healthcare FL work\"}},\n"
    " {{\"id\": \"s2:y2\", \"score\": 1, \"rationale\": \"Popular benchmark dataset, not related to federated learning or healthcare\"}}]\n\n"
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


# ── Scholar-First Pipeline: Scout Queries ─────────────────────────────

SCOUT_QUERIES = (
    "You are an expert research librarian. Given a user's research query, "
    "generate exactly 3 diverse search queries optimized for Google Scholar "
    "(which searches the full text of papers, not just titles and abstracts).\n\n"
    "Strategy:\n"
    "- Query A (component decomposition): Break the query into its independent "
    "sub-concepts and search for them separately or in combination. Use quoted "
    "phrases for specific technical terms.\n"
    "- Query B (broader/adjacent framing): Rephrase using alternative "
    "terminology, parent concepts, or adjacent methods that might be used "
    "in papers addressing the same problem.\n"
    "- Query C (acronym + application): Use common acronyms, abbreviations, "
    "and the application domain together.\n\n"
    "Important:\n"
    "- If you are uncertain about the exact terminology in this field, lean "
    "towards BROADER queries that cast a wider net.\n"
    "- Always include both full forms and common abbreviations/acronyms.\n"
    "- Google Scholar supports quoted phrases for exact matching.\n\n"
    "User Query: {query}\n\n"
    "Return ONLY a JSON object:\n"
    '{{"queries": ["query A text", "query B text", "query C text"]}}'
)


# ── Scholar-First Pipeline: Discovery Read ────────────────────────────

DISCOVERY_READ = (
    "You are analyzing search results to discover the real "
    "vocabulary and key players in a research area.\n\n"
    "User's research query: {query}\n\n"
    "=== Google Scholar results (academic papers) ===\n{scholar_results}\n\n"
    "=== Google Web results (blogs, tutorials, discussions) ===\n{web_results}\n\n"
    "IMPORTANT: Web results may contain blog posts, tutorials, GitHub pages, "
    "and informal discussions. These are valuable for discovering alternative "
    "terminology, system names, and community jargon that papers don't use in "
    "their titles. However, IGNORE any web results that are clearly unrelated "
    "to the research query (ads, unrelated products, etc.).\n\n"
    "Your task:\n"
    "1. Extract specific technical terms, system names, method names, and "
    "acronyms that appear in EITHER Scholar or Web results and are relevant "
    "to the query but were NOT in the original query. Web results are "
    "especially useful for discovering informal names, project names, and "
    "alternative phrasings used by the research community.\n"
    "2. Identify author names that appear across multiple results (domain experts).\n"
    "3. Generate 5-7 search query angles using the discovered vocabulary. "
    "For EACH angle, provide a tailored query for each academic API:\n"
    "  - semantic_scholar: natural language sentence (embedding-based search, full sentences work best)\n"
    "  - arxiv: short 2-5 word keyword phrase\n"
    "  - openalex: keyword query with the most important terms\n\n"
    "Make each angle genuinely different. Include at least one angle that "
    "searches for specific system/method names found in the snippets, and "
    "one that targets a key author's work.\n\n"
    "Return ONLY a JSON object:\n"
    '{{\n'
    '  "discovered_terms": ["term1", "term2", ...],\n'
    '  "discovered_authors": ["Author Name", ...],\n'
    '  "query_angles": [\n'
    '    {{\n'
    '      "label": "short description of this angle",\n'
    '      "semantic_scholar": "...",\n'
    '      "arxiv": "...",\n'
    '      "openalex": "..."\n'
    '    }}\n'
    '  ]\n'
    '}}'
)


# ── Scholar-First Pipeline: Re-Rank Pass ─────────────────────────────

RERANK_PASS = (
    "You previously scored these papers individually. Now compare them "
    "against each other for a more nuanced ranking.\n\n"
    "QUERY: {query}\n\n"
    "PAPERS (already scored >= 6 in first pass):\n{papers_json}\n\n"
    "Re-evaluate each paper's relevance considering the full set. "
    "Some papers may deserve a higher score now that you see the full "
    "landscape of results. Focus on:\n"
    "- Papers that are clearly more central than their initial score suggests\n"
    "- Papers that are near-duplicates (lower the weaker one)\n"
    "- Methodological papers vs survey/overview papers (methods > surveys for the same score)\n\n"
    "Return a JSON array with updated scores:\n"
    '[{{"id": "...", "score": 8, "rationale": "one-sentence reason"}}]\n'
    "Return ONLY the JSON array, no other text."
)


# ── Multi-hop: Seed Selection ─────────────────────────────────────────

SEED_SELECTION = (
    "You are selecting papers whose citation neighborhoods are worth exploring.\n\n"
    "QUERY: {query}\n\n"
    "CANDIDATE PAPERS (top results so far):\n{papers_json}\n\n"
    "Select up to 15 papers that are most likely to have valuable citation "
    "neighborhoods. Prefer:\n"
    "- Original method/system papers (their references and citations trace the idea chain)\n"
    "- Papers at the intersection of two relevant subfields (bridge papers)\n"
    "- Recent papers with high citation velocity (their citations are the active frontier)\n\n"
    "Avoid:\n"
    "- Broad survey papers (their citations are too diverse to be useful)\n"
    "- Very old foundational papers (their citation neighborhoods are too large)\n"
    "- Tangentially related papers (their neighborhoods will drift off-topic)\n\n"
    "Return ONLY a JSON array of paper IDs to explore:\n"
    '[{{"id": "...", "reason": "one-sentence reason"}}]\n'
    "Return ONLY the JSON array."
)


# ── Multi-hop: Drift Check ────────────────────────────────────────────

HOP_DRIFT_CHECK = (
    "You are checking whether citation graph expansion is still on-topic.\n\n"
    "ORIGINAL QUERY: {query}\n\n"
    "Papers discovered in hop 1 (titles only):\n{hop1_titles}\n\n"
    "Question: Are these discovered papers still aligned with the original query? "
    "If most papers are tangential or from a different field, the expansion has drifted.\n\n"
    "Return ONLY a JSON object:\n"
    '{{"aligned": true/false, "drift_score": 0.0-1.0, "reason": "one sentence"}}\n'
    "drift_score: 0.0 = perfectly on-topic, 1.0 = completely off-topic.\n"
    "If drift_score > 0.6, we will skip further hops."
)


# ── Deep Rank: Final Ordering ─────────────────────────────────────────

FINAL_ORDERING = (
    "You are producing the final ranking of papers for a researcher.\n\n"
    "QUERY: {query}\n\n"
    "TOP PAPERS with all scoring signals:\n{papers_with_signals}\n\n"
    "Produce a final ranking considering ALL signals holistically:\n"
    "- llm_relevance: your previous relevance judgment (0-10)\n"
    "- citation_velocity: citations per year (higher = more impactful)\n"
    "- in_set_citations: how many other results cite this paper (higher = more foundational)\n"
    "- influential_citations: S2 influential citation count (higher = methodologically important)\n"
    "- author_reputation: author overlap with other results (higher = domain expert)\n"
    "- venue: publication venue quality\n\n"
    "A paper with moderate LLM score but very high in-set citations might deserve a top spot "
    "(it's foundational). A paper with high LLM score but zero citations in the last 3 years "
    "might be less useful than it appears.\n\n"
    "Return a JSON array of paper IDs in your recommended order (best first):\n"
    '[{{"id": "...", "final_score": 9.2, "rationale": "brief reason"}}]\n'
    "Return ONLY the JSON array."
)


# ── Final Slate: Diversity Judge ──────────────────────────────────────

DIVERSITY_SLATE = (
    "You are selecting the final set of papers to present to a researcher.\n\n"
    "QUERY: {query}\n\n"
    "RANKED PAPERS (already in quality order):\n{papers_json}\n\n"
    "Select the best 20-{max_results} papers ensuring DIVERSITY:\n"
    "- Include a mix: foundational papers, recent advances, methods papers, application papers\n"
    "- Remove near-duplicates (papers covering essentially the same contribution)\n"
    "- NEVER drop a paper scoring 8+ in favor of a lower-scoring paper purely for diversity\n"
    "- Only replace papers scoring 6-7 that are redundant with a higher-scoring paper\n"
    "- Preserve the top 5 papers unconditionally (they are anchor papers)\n\n"
    "Return a JSON array of selected paper IDs in final order:\n"
    '["paper_id_1", "paper_id_2", ...]\n'
    "Return ONLY the JSON array."
)


# ── Explore Citation Graph: Relevance Score ───────────────────────────

EXPLORE_GRAPH_SCORE = (
    "You are evaluating papers discovered in the citation neighborhood of a seed paper.\n\n"
    "SEED PAPER:\n"
    "Title: {seed_title}\n"
    "Abstract: {seed_abstract}\n\n"
    "DISCOVERED PAPERS:\n{papers_json}\n\n"
    "Score each paper 0-10 for how relevant/useful it is in the context of "
    "the seed paper's research area. Prefer papers that:\n"
    "- Build on, extend, or improve the seed paper's approach\n"
    "- Provide key foundational methods used by the seed paper\n"
    "- Address the same problem from a different angle\n"
    "- Provide important evaluation benchmarks or datasets\n\n"
    "Return a JSON array:\n"
    '[{{"id": "...", "score": 7, "rationale": "one-sentence reason"}}]\n'
    "Return ONLY the JSON array."
)
