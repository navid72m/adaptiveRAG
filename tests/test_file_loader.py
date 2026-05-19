"""Unit tests for file_loader — uses tmp_path, no Ollama required."""
import json
import os
import pytest

from adaptiverag.pipeline.file_loader import (
    load_documents,
    load_validation_set,
    _read_md,
    _read_txt,
)


# ── load_validation_set ────────────────────────────────────────────────────

class TestLoadValidationSet:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load_validation_set(path) == []

    def test_does_not_create_file_when_missing(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        load_validation_set(path)
        assert not os.path.exists(path)

    def test_returns_parsed_json(self, tmp_path):
        data = [{"query": "What is X?", "expected_answer": "X is Y."}]
        p = tmp_path / "val.json"
        p.write_text(json.dumps(data))
        assert load_validation_set(str(p)) == data

    def test_returns_multiple_entries(self, tmp_path):
        data = [
            {"query": "Q1", "expected_answer": "A1"},
            {"query": "Q2", "expected_answer": "A2"},
        ]
        p = tmp_path / "val.json"
        p.write_text(json.dumps(data))
        result = load_validation_set(str(p))
        assert len(result) == 2


# ── _read_txt ──────────────────────────────────────────────────────────────

class TestReadTxt:
    def test_reads_plain_text(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("Hello world", encoding="utf-8")
        assert _read_txt(str(p)) == "Hello world"

    def test_handles_utf8(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("café résumé", encoding="utf-8")
        assert "café" in _read_txt(str(p))


# ── _read_md ───────────────────────────────────────────────────────────────

class TestReadMd:
    def test_strips_headings(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("# Title\n## Sub\nBody text", encoding="utf-8")
        result = _read_md(str(p))
        assert "#" not in result
        assert "Title" in result
        assert "Body text" in result

    def test_strips_bold_and_italic(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("**bold** and _italic_ text", encoding="utf-8")
        result = _read_md(str(p))
        assert "**" not in result
        assert "_" not in result
        assert "bold" in result
        assert "italic" in result

    def test_strips_inline_links(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("[click here](https://example.com)", encoding="utf-8")
        result = _read_md(str(p))
        assert "https://example.com" not in result
        assert "click here" in result

    def test_strips_code_fences(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("```python\nprint('hi')\n```\nAfter", encoding="utf-8")
        result = _read_md(str(p))
        assert "```" not in result
        assert "After" in result

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "doc.md"
        p.write_text("", encoding="utf-8")
        assert _read_md(str(p)) == ""


# ── load_documents ─────────────────────────────────────────────────────────

class TestLoadDocuments:
    def test_loads_txt_file(self, tmp_path):
        (tmp_path / "a.txt").write_text("content here", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["filename"] == "a.txt"
        assert docs[0]["content"] == "content here"
        assert docs[0]["ext"] == "txt"

    def test_loads_md_file(self, tmp_path):
        (tmp_path / "b.md").write_text("# Title\nBody", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 1
        assert docs[0]["filename"] == "b.md"

    def test_skips_unsupported_extensions(self, tmp_path):
        (tmp_path / "file.csv").write_text("a,b,c", encoding="utf-8")
        (tmp_path / "file.xml").write_text("<x/>", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("   \n\t  ", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert docs == []

    def test_loads_multiple_files_sorted(self, tmp_path):
        (tmp_path / "b.txt").write_text("B content", encoding="utf-8")
        (tmp_path / "a.txt").write_text("A content", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert len(docs) == 2
        assert docs[0]["filename"] == "a.txt"
        assert docs[1]["filename"] == "b.txt"

    def test_returns_empty_for_missing_folder(self, tmp_path):
        docs = load_documents(str(tmp_path / "nonexistent"))
        assert docs == []

    def test_doc_has_required_keys(self, tmp_path):
        (tmp_path / "doc.txt").write_text("some text", encoding="utf-8")
        docs = load_documents(str(tmp_path))
        assert set(docs[0].keys()) >= {"filename", "content", "path", "ext"}
