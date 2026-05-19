"""
Feed-kind workspaces: a self-improving recommender for ongoing reading material.

Distinct from `research`-kind workspaces (loom's original): instead of a
prompt-driven seed search + multi-hop citation expansion, a feed workspace
operates on demand -- `feed_more(n, window)` finds N new relevant items
from configured sources, ranks them, and surfaces a digest. Ratings feed
back into the ranker, which evolves through three stages as labels accumulate.
"""
