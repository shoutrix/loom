"""
Workspace Brief — the auto-generated context document for a workspace.

Each workspace has a per-workspace brief that captures its goal, scope,
key questions, current focus, and exclusions. The brief is:
  - auto-generated when the workspace first reaches 3 ingested papers;
  - re-generated every 20 *additional* ingested papers afterwards;
  - editable in the UI alongside a separate ``user_notes`` field that
    is preserved across regenerations.

Agents are encouraged to call ``get_workspace_brief`` when they begin
working on a workspace — it tells them the goal and scope so they can
make sane categorization decisions and filter candidates accordingly.

On-disk schema (data/<workspace>/workspace_brief.json):

    {
      "version": 1,
      "generated_at": "...",
      "generated_from_paper_count": 23,
      "model": "gemini-2.5-pro",
      "brief": {
        "goal": "...",
        "scope": "...",
        "key_questions": ["...", "..."],
        "current_focus": "...",
        "exclude": "..."
      },
      "user_notes": "free-form markdown"
    }
"""

from __future__ import annotations

from loom.workspace_brief.brief import (
    AUTO_REGEN_EVERY_N_INGESTED,
    MIN_INGESTED_FOR_FIRST_BRIEF,
    Brief,
    BriefDocument,
    brief_path,
    generate_brief,
    load_brief,
    save_brief,
    should_regenerate,
)

__all__ = [
    "AUTO_REGEN_EVERY_N_INGESTED",
    "MIN_INGESTED_FOR_FIRST_BRIEF",
    "Brief",
    "BriefDocument",
    "brief_path",
    "generate_brief",
    "load_brief",
    "save_brief",
    "should_regenerate",
]
