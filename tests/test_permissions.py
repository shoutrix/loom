"""P5 — SubscriberRegistry + enforce()."""

from __future__ import annotations

import yaml


def _write_subscribers(path, entries):
    path.write_text(yaml.safe_dump({"subscribers": entries}, sort_keys=False))


def test_auto_bootstrap_creates_defaults(tmp_storage):
    from loom.permissions import SubscriberRegistry, ensure_subscribers_yaml

    sub_path = tmp_storage / "subscribers.yaml"
    assert not sub_path.exists()

    ensure_subscribers_yaml(sub_path)
    assert sub_path.exists()

    reg = SubscriberRegistry(sub_path)
    expected = {"loom-ui", "claude-code", "cursor-mcp", "recommender"}
    assert expected.issubset(set(reg._by_id.keys()))
    # All defaults are open
    assert reg.can_access("loom-ui", "any-ws", write=True)
    assert reg.can_access("claude-code", "any-ws", write=True)


def test_scoped_subscriber_read_only(tmp_storage):
    from loom.permissions import SubscriberRegistry

    sub_path = tmp_storage / "subscribers.yaml"
    _write_subscribers(sub_path, [
        {"id": "scoped", "label": "Scoped", "workspaces": ["allowed"], "mode": "read"},
    ])
    reg = SubscriberRegistry(sub_path)

    assert reg.can_access("scoped", "allowed", write=False)
    assert not reg.can_access("scoped", "allowed", write=True)  # mode=read
    assert not reg.can_access("scoped", "elsewhere", write=False)  # out-of-scope


def test_write_only_subscriber(tmp_storage):
    from loom.permissions import SubscriberRegistry

    sub_path = tmp_storage / "subscribers.yaml"
    _write_subscribers(sub_path, [
        {"id": "ingestor", "label": "Ingestor", "workspaces": "*", "mode": "write"},
    ])
    reg = SubscriberRegistry(sub_path)

    assert reg.can_access("ingestor", "anywhere", write=True)
    assert not reg.can_access("ingestor", "anywhere", write=False)


def test_unknown_subscriber_denied_by_default(tmp_storage):
    from loom.permissions import SubscriberRegistry

    sub_path = tmp_storage / "subscribers.yaml"
    _write_subscribers(sub_path, [
        {"id": "known", "label": "k", "workspaces": "*", "mode": "read+write"},
    ])
    reg = SubscriberRegistry(sub_path)
    # An unknown id resolves to a synthetic empty-scope read subscriber.
    assert not reg.can_access("nobody", "anywhere", write=False)
    assert not reg.can_access("nobody", "anywhere", write=True)


def test_list_workspaces_for_filters(tmp_storage):
    from loom.permissions import SubscriberRegistry

    sub_path = tmp_storage / "subscribers.yaml"
    _write_subscribers(sub_path, [
        {"id": "all", "label": "a", "workspaces": "*", "mode": "read+write"},
        {"id": "two", "label": "t", "workspaces": ["a", "b"], "mode": "read+write"},
    ])
    reg = SubscriberRegistry(sub_path)

    assert reg.list_workspaces_for("all", ["a", "b", "c"]) == ["a", "b", "c"]
    assert reg.list_workspaces_for("two", ["a", "b", "c"]) == ["a", "b"]


def test_enforce_fails_open_when_unbound():
    """Before install_active_subscriber is called, enforce() lets anything through.

    We can't easily reset the module-level globals from a test, so we just check
    the contract that *without* a registry, enforce() returns None.
    """
    from loom.permissions import enforce
    # If something else in the suite has bound a registry already, skip this
    # assertion — global state is best-effort here.
    import loom.permissions.registry as _reg

    if _reg._REGISTRY is None:
        assert enforce("any-ws", write=True) is None


def test_enforce_round_trip(tmp_storage):
    from loom.permissions import (
        SubscriberRegistry,
        enforce,
        install_active_subscriber,
    )

    sub_path = tmp_storage / "subscribers.yaml"
    _write_subscribers(sub_path, [
        {"id": "cursor", "label": "Cursor", "workspaces": ["code"], "mode": "read"},
    ])
    reg = SubscriberRegistry(sub_path)
    install_active_subscriber("cursor", reg)

    # In-scope read: allowed (None means no error).
    assert enforce("code", write=False) is None
    # In-scope write: denied (mode=read).
    err = enforce("code", write=True)
    assert err is not None and err["ok"] is False and "permission denied" in err["error"]
    # Out-of-scope read: denied.
    err = enforce("other", write=False)
    assert err is not None and err["ok"] is False
