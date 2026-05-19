"""
Personalized recommender service.

Under the unified workspace model (P6+), the recommender is no longer a
distinct workspace kind. Any workspace can have a recommender attached:
the presence of <data_dir>/<workspace>/feed.db marks it as recommender-
enabled. The package was originally called `feed` and was renamed to
`recommender` when the kind flag was dropped.

Key entry points:
- service.RecommenderService — orchestrator over the in-package primitives.
- pipeline.run_more — fetch + score + persist a batch of candidates.
- ranker.refit — fit the appropriate stage (0/1/2) based on rating count.
- profile.get_profile / centroids — per-workspace profile + centroid state.

The on-disk SQLite db keeps its filename (feed.db) to avoid migrating
existing workspaces; only the package import path changed.
"""
