from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pypdf

from ..core.config import VAL_QUERIES_PATH

try:
    import docx as python_docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


def _read_txt(path: str) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()


def _read_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "".join(pg.extract_text() or "" for pg in reader.pages)


def _read_md(path: str) -> str:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    text = re.sub(r'```[\w]*\n', '', text)
    text = re.sub(r'```', '', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return re.sub(r'[*_`~]', '', text)


def _read_docx(path: str) -> str:
    if not _DOCX_AVAILABLE:
        return ""
    return "\n".join(
        p.text for p in python_docx.Document(path).paragraphs if p.text.strip()
    )


_READERS = {
    ".txt":  _read_txt,
    ".pdf":  _read_pdf,
    ".md":   _read_md,
    ".docx": _read_docx,
}


def load_documents(folder: str) -> List[Dict]:
    docs = []
    for f in sorted(glob.glob(os.path.join(folder, "*.*"))):
        ext    = Path(f).suffix.lower()
        reader = _READERS.get(ext)
        if not reader:
            continue
        try:
            content = reader(f)
        except Exception as e:
            print(f"  ✗   Couldn't read {f}: {e}")
            continue
        if content.strip():
            docs.append({
                "filename": os.path.basename(f),
                "content":  content,
                "path":     f,
                "ext":      ext.lstrip("."),
            })
    return docs


def load_validation_set(path: str = VAL_QUERIES_PATH) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fp:
        return json.load(fp)
