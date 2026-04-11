"""
Document parsers: convert various formats to markdown.

- MarkItDown: PDF, PPTX, EPUB, HTML, DOCX, images
- arxiv-txt.org: plain-text extraction for arXiv papers
- youtube-transcript-api: YouTube video transcripts
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    content: str
    source_type: str  # "file", "arxiv", "youtube", "url", "text"
    source_url: str = ""
    abstract: str = ""


def parse_file(file_path: str | Path) -> ParsedDocument:
    """Parse a local file using MarkItDown."""
    from markitdown import MarkItDown

    path = Path(file_path)
    md = MarkItDown()
    result = md.convert(str(path))
    content = result.text_content or ""
    content = filter_key_sections(content)
    title = _extract_title(content) or path.stem

    import hashlib
    doc_id = hashlib.sha256(content[:2000].encode()).hexdigest()[:16]

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        content=content,
        source_type="file",
        source_url=str(path),
        abstract=_extract_abstract(content),
    )


def parse_arxiv(arxiv_id: str) -> ParsedDocument:
    """Fetch full text from multiple sources and fall back to abstract."""
    arxiv_id = arxiv_id.strip().replace("https://arxiv.org/abs/", "")
    arxiv_id = arxiv_id.replace("https://arxiv.org/pdf/", "").replace(".pdf", "")

    full_text_urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv-txt.org/raw/{arxiv_id}/txt",
    ]

    for url in full_text_urls:
        try:
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                continue

            if "html" in url:
                from markitdown import MarkItDown
                import io
                md = MarkItDown()
                result = md.convert_stream(
                    io.BytesIO(resp.content), file_extension=".html"
                )
                content = result.text_content or ""
            else:
                content = resp.text

            if len(content) > 500:
                content = _clean_arxiv_html_content(content)
                content = filter_key_sections(content)
                title = _extract_title(content) or f"arXiv:{arxiv_id}"
                return ParsedDocument(
                    doc_id=f"arxiv_{arxiv_id.replace('/', '_').replace('.', '_')}",
                    title=title,
                    content=content,
                    source_type="arxiv",
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    abstract=_extract_abstract(content),
                )
        except Exception:
            continue

    abs_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        resp = requests.get(abs_url, timeout=15)
        if resp.status_code == 200:
            import xml.etree.ElementTree as ET
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(resp.text)
            entry = root.find("atom:entry", ns)
            if entry is not None:
                title = (entry.findtext("atom:title", "", ns) or "").strip()
                abstract = (entry.findtext("atom:summary", "", ns) or "").strip()
                content = f"# {title}\n\n## Abstract\n\n{abstract}"
                return ParsedDocument(
                    doc_id=f"arxiv_{arxiv_id.replace('/', '_').replace('.', '_')}",
                    title=title,
                    content=content,
                    source_type="arxiv",
                    source_url=f"https://arxiv.org/abs/{arxiv_id}",
                    abstract=abstract,
                )
    except Exception:
        pass

    raise ValueError(f"Could not fetch arXiv paper: {arxiv_id}")


def parse_youtube(url: str) -> ParsedDocument:
    """Extract transcript from a YouTube video."""
    video_id = _extract_youtube_id(url)
    if not video_id:
        raise ValueError(f"Could not extract YouTube video ID from: {url}")

    from youtube_transcript_api import YouTubeTranscriptApi

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    lines = [entry["text"] for entry in transcript_list]
    content = " ".join(lines)

    title = f"YouTube: {video_id}"
    try:
        page = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
        match = re.search(r"<title>(.*?)</title>", page.text)
        if match:
            title = match.group(1).replace(" - YouTube", "").strip()
    except Exception:
        pass

    import hashlib
    doc_id = f"yt_{video_id}"

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        content=f"# {title}\n\n{content}",
        source_type="youtube",
        source_url=url,
        abstract=content[:500],
    )


def parse_url(url: str) -> ParsedDocument:
    """Parse a web URL using MarkItDown."""
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(url)
    content = result.text_content or ""
    content = filter_key_sections(content)
    title = _extract_title(content) or url

    import hashlib
    doc_id = hashlib.sha256(url.encode()).hexdigest()[:16]

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        content=content,
        source_type="url",
        source_url=url,
        abstract=_extract_abstract(content),
    )


def parse_text(text: str, title: str = "Untitled Note") -> ParsedDocument:
    """Wrap raw text as a parsed document."""
    import hashlib
    doc_id = hashlib.sha256(text[:2000].encode()).hexdigest()[:16]
    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        content=text,
        source_type="text",
        abstract=text[:500],
    )


def _extract_title(content: str) -> str:
    for line in content.split("\n")[:20]:
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _clean_arxiv_html_content(content: str) -> str:
    """Strip arXiv HTML boilerplate from MarkItDown conversion."""
    lines = content.split("\n")

    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if stripped.startswith("# ") and len(stripped) > 4:
            start_idx = i
            break
        if "abstract" in stripped and (stripped.startswith("#") or stripped.startswith("**")):
            start_idx = max(0, i - 2)
            break

    end_idx = len(lines)
    footer_markers = [
        "generated by latexml", "beta", "convert latex packages",
        "developer contributions", "why html?", "report github issue",
        "accessibility forum", "arxiv accessibility",
    ]
    for i in range(len(lines) - 1, max(end_idx - 50, 0), -1):
        stripped = lines[i].strip().lower()
        if any(marker in stripped for marker in footer_markers):
            end_idx = i
            break

    cleaned = "\n".join(lines[start_idx:end_idx]).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


_KEY_SECTION_PATTERNS = [
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:abstract)\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:introduction)\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:method(?:ology|s)?|approach|proposed\s+(?:method|approach|framework))\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:model|architecture|framework|system\s+(?:design|overview))\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:experiment(?:s|al)?|results?|evaluation|empirical)\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:discussion)\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:conclusion(?:s)?|summary)\b", re.IGNORECASE),
    re.compile(r"^#{1,3}\s*(?:\d+\.?\s*)?(?:related\s+work|background|prior\s+work|literature\s+review)\b", re.IGNORECASE),
]

CHARS_PER_PAGE = 3000


def filter_key_sections(content: str) -> str:
    """For long papers (>5 pages), keep only key sections."""
    estimated_pages = len(content) / CHARS_PER_PAGE
    if estimated_pages <= 5:
        return content

    lines = content.split("\n")
    heading_re = re.compile(r"^(#{1,3})\s+(.+)$")

    sections: list[tuple[int, str, str]] = []
    for i, line in enumerate(lines):
        m = heading_re.match(line)
        if m:
            sections.append((i, m.group(1), m.group(2)))

    if not sections:
        return content

    keep_ranges: list[tuple[int, int]] = []
    # Always keep the preamble (title area, before first heading)
    if sections[0][0] > 0:
        keep_ranges.append((0, sections[0][0]))

    for idx, (line_no, level, heading_text) in enumerate(sections):
        is_key = any(pat.match(f"{level} {heading_text}") for pat in _KEY_SECTION_PATTERNS)
        if is_key:
            end = sections[idx + 1][0] if idx + 1 < len(sections) else len(lines)
            keep_ranges.append((line_no, end))

    if not keep_ranges:
        return content

    kept_lines: list[str] = []
    for start, end in keep_ranges:
        kept_lines.extend(lines[start:end])
        kept_lines.append("")

    result = "\n".join(kept_lines).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _extract_abstract(content: str) -> str:
    content_lower = content.lower()
    for marker in ["## abstract", "abstract:", "abstract\n"]:
        idx = content_lower.find(marker)
        if idx >= 0:
            start = idx + len(marker)
            chunk = content[start:start + 1000].strip()
            end = chunk.find("\n\n")
            if end > 0:
                return chunk[:end].strip()
            return chunk[:500].strip()
    return content[:500].strip()


def _extract_youtube_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None
