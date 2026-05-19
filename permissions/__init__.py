"""Subscriber-scoped permissions for the knowledge substrate.

Each consumer of the knowledge base (Loom UI, Claude Code MCP client,
Cursor MCP client, the recommender service, …) is registered as a
Subscriber with a declared workspace scope and read/write mode. MCP tools
and the FastAPI workspace routes enforce the scope at request time.

The registry is just a YAML file under storage_root_dir/subscribers.yaml.
It's auto-generated on first start with permissive defaults so nothing
breaks; tighten it after the system is up.
"""

from __future__ import annotations

from loom.permissions.registry import (
    Subscriber,
    SubscriberRegistry,
    active_subscriber,
    active_subscriber_id,
    ensure_subscribers_yaml,
    enforce,
    install_active_subscriber,
)

__all__ = [
    "Subscriber",
    "SubscriberRegistry",
    "active_subscriber",
    "active_subscriber_id",
    "ensure_subscribers_yaml",
    "enforce",
    "install_active_subscriber",
]
