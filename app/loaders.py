
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.config import PROCESSED_DIR


@dataclass
class Section:
    heading: str
    text: str
    page: int | None = None


@dataclass
class LoadedDocument:
    source: str          # file name
    file_type: str       # md / txt / html / pdf
    sections: list[Section]


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_markdown(path: Path) -> list[Section]:
    sections: list[Section] = []
    heading, buf, in_code = "", [], False

    def flush():
        body = _clean("\n".join(buf))
        if body:
            sections.append(Section(heading, body))

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADING_RE.match(line)
        if m:
            flush()
            buf.clear()
            heading = m.group(2)
        else:
            buf.append(line)
    flush()
    return sections


def load_text(path: Path) -> list[Section]:
    body = _clean(path.read_text(encoding="utf-8", errors="ignore"))
    return [Section("", body)] if body else []


def load_html(path: Path) -> list[Section]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    root = soup.body or soup
    heading = soup.title.get_text(strip=True) if soup.title else ""
    sections: list[Section] = []
    buf: list[str] = []

    def flush():
        body = _clean("\n\n".join(buf))
        if body:
            sections.append(Section(heading, body))

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    block_tags = ["p", "li", "pre", "blockquote"]
    for el in root.find_all(list(heading_tags) + block_tags):
        if el.name in heading_tags:
            flush()
            buf.clear()
            heading = el.get_text(" ", strip=True)
        else:
            if el.find(block_tags):      # skip containers; their children get visited
                continue
            text = el.get_text(" ", strip=True)
            if text:
                buf.append(text)
    flush()
    return sections


def load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(str(path))
    sections = []
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            sections.append(Section("", text, page=i))
    return sections


LOADERS = {
    ".md": ("md", load_markdown),
    ".markdown": ("md", load_markdown),
    ".txt": ("txt", load_text),
    ".html": ("html", load_html),
    ".htm": ("html", load_html),
    ".pdf": ("pdf", load_pdf),
}


def load_document(path: Path) -> LoadedDocument:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    file_type, fn = LOADERS[ext]
    return LoadedDocument(source=path.name, file_type=file_type, sections=fn(path))


def save_processed(doc: LoadedDocument) -> Path:
    out = PROCESSED_DIR / f"{doc.source}.json"
    out.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
    return out