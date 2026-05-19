"""Source adapter base + window helpers."""

from __future__ import annotations

import datetime
from typing import Iterable, Protocol

from loom.feed.models import Candidate


class SourceAdapter(Protocol):
    """A source adapter fetches candidates matching a query and time window."""

    name: str

    def fetch(
        self,
        query: str,
        *,
        since: datetime.datetime | None,
        until: datetime.datetime | None,
        limit: int = 50,
    ) -> list[Candidate]: ...


def parse_window(
    window: str,
    last_run_at: str | None = None,
) -> tuple[datetime.datetime | None, datetime.datetime | None]:
    """
    Parse a window string into (since, until) datetimes.

    Accepted forms:
      'all'             -> (None, None)
      'since-last-run'  -> (last_run_at, now)  -- None if no last run
      '<n>d'/'1w'       -> (now - n days, now)
      '<n>m'            -> (now - n months, now)  (approximate, 30d/month)
      '<n>y'            -> (now - n years, now)
    """
    now = datetime.datetime.now(datetime.UTC)
    if not window or window == "all":
        return None, None
    if window == "since-last-run":
        if not last_run_at:
            return None, now
        try:
            since = datetime.datetime.fromisoformat(last_run_at.replace("Z", "+00:00"))
            if since.tzinfo is None:
                since = since.replace(tzinfo=datetime.UTC)
        except Exception:
            return None, now
        return since, now

    unit = window[-1].lower()
    try:
        n = int(window[:-1])
    except ValueError:
        return None, now

    if unit == "d":
        return now - datetime.timedelta(days=n), now
    if unit == "w":
        return now - datetime.timedelta(weeks=n), now
    if unit == "m":
        return now - datetime.timedelta(days=30 * n), now
    if unit == "y":
        return now - datetime.timedelta(days=365 * n), now
    return None, now
