"""
Local vault folder management (Obsidian-like).

Provides create/list/read/write operations on markdown files in a
structured directory. The vault is the source of truth for raw content
and is designed to be synced to Google Drive.
"""

from __future__ import annotations

import re
import datetime
from pathlib import Path
from dataclasses import dataclass


@dataclass
class VaultFile:
    path: Path
    relative_path: str
    title: str
    size_bytes: int
    modified: datetime.datetime


class VaultManager:
    """Manages a local directory of markdown files."""

    def __init__(self, vault_dir: Path) -> None:
        self.root = vault_dir
        self.root.mkdir(parents=True, exist_ok=True)

    def list_files(self, subfolder: str = "") -> list[VaultFile]:
        target = self.root / subfolder if subfolder else self.root
        if not target.exists():
            return []
        files: list[VaultFile] = []
        for p in sorted(target.rglob("*.md")):
            files.append(self._to_vault_file(p))
        return files

    def read_file(self, relative_path: str) -> str | None:
        fp = self.root / relative_path
        if not fp.exists() or not fp.is_file():
            return None
        return fp.read_text(encoding="utf-8")

    def write_file(self, relative_path: str, content: str) -> VaultFile:
        fp = self.root / relative_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return self._to_vault_file(fp)

    def create_note(
        self,
        title: str,
        content: str,
        subfolder: str = "notes",
    ) -> VaultFile:
        slug = _slugify(title)
        ts = datetime.datetime.now().strftime("%Y%m%d")
        relative = f"{subfolder}/{ts}_{slug}.md"

        frontmatter = f"---\ntitle: {title}\ncreated: {datetime.datetime.now().isoformat()}\n---\n\n"
        full_content = frontmatter + content
        return self.write_file(relative, full_content)

    def save_ingested_document(
        self,
        doc_id: str,
        title: str,
        markdown_content: str,
        source_type: str,
        source_url: str = "",
    ) -> VaultFile:
        slug = _slugify(title)[:60]
        relative = f"ingested/{source_type}/{slug}_{doc_id[:8]}.md"

        frontmatter_lines = [
            "---",
            f"title: \"{title}\"",
            f"doc_id: {doc_id}",
            f"source_type: {source_type}",
        ]
        if source_url:
            frontmatter_lines.append(f"source_url: {source_url}")
        frontmatter_lines.append(f"ingested: {datetime.datetime.now().isoformat()}")
        frontmatter_lines.append("---\n")

        full_content = "\n".join(frontmatter_lines) + "\n" + markdown_content
        return self.write_file(relative, full_content)

    def delete_file(self, relative_path: str) -> bool:
        fp = self.root / relative_path
        if fp.exists():
            fp.unlink()
            return True
        return False

    def _to_vault_file(self, path: Path) -> VaultFile:
        stat = path.stat()
        stem = path.stem
        title = stem.replace("_", " ").replace("-", " ")
        return VaultFile(
            path=path,
            relative_path=str(path.relative_to(self.root)),
            title=title,
            size_bytes=stat.st_size,
            modified=datetime.datetime.fromtimestamp(stat.st_mtime),
        )


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s]+", "_", text)
    return text[:80]
