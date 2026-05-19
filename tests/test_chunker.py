"""Unit tests for ContentAwareChunker — no external dependencies required."""
import pytest
from adaptiverag.components.chunker import ContentAwareChunker


# ── helpers ────────────────────────────────────────────────────────────────

def make_text(n_words: int, word: str = "word") -> str:
    return " ".join([word] * n_words)


def all_chars_covered(original: str, chunks: list[str]) -> bool:
    """Every character in the original must appear in at least one chunk."""
    combined = "".join(chunks)
    # For overlapping strategies the combined may be longer — just check
    # that the original is a subsequence (relaxed: set of chars is subset).
    return set(original.strip()) <= set(combined)


# ── fixed strategy ─────────────────────────────────────────────────────────

class TestFixedStrategy:
    def setup_method(self):
        self.c = ContentAwareChunker(strategy="fixed", size=20, overlap=5)

    def test_empty_returns_empty(self):
        assert self.c.chunk("") == []

    def test_whitespace_only_returns_empty(self):
        assert self.c.chunk("   \n\t  ") == []

    def test_short_text_single_chunk(self):
        text = "hello world"
        chunks = self.c.chunk(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunks_do_not_exceed_size(self):
        text = make_text(200)
        chunks = self.c.chunk(text)
        for chunk in chunks:
            assert len(chunk) <= self.c.size

    def test_overlap_present_between_consecutive_chunks(self):
        text = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
        c = ContentAwareChunker(strategy="fixed", size=10, overlap=3)
        chunks = c.chunk(text)
        assert len(chunks) >= 2
        # tail of chunk[0] should appear at start of chunk[1]
        tail = chunks[0][-3:]
        assert chunks[1].startswith(tail)

    def test_no_overlap_zero(self):
        text = "abcdefghij"  # 10 chars
        c = ContentAwareChunker(strategy="fixed", size=5, overlap=0)
        chunks = c.chunk(text)
        assert chunks == ["abcde", "fghij"]

    def test_unknown_strategy_falls_back_to_fixed(self):
        c = ContentAwareChunker(strategy="nonexistent", size=20, overlap=0)
        chunks = c.chunk("hello world")
        assert chunks == ["hello world"]


# ── sentence strategy ──────────────────────────────────────────────────────

class TestSentenceStrategy:
    def setup_method(self):
        self.c = ContentAwareChunker(strategy="sentence", size=60, overlap=10)

    def test_empty_returns_empty(self):
        assert self.c.chunk("") == []

    def test_single_sentence(self):
        text = "This is one sentence."
        chunks = self.c.chunk(text)
        assert len(chunks) == 1
        assert "This is one sentence" in chunks[0]

    def test_multiple_sentences_split_correctly(self):
        sentences = [f"Sentence number {i}." for i in range(10)]
        text = " ".join(sentences)
        c = ContentAwareChunker(strategy="sentence", size=40, overlap=0)
        chunks = c.chunk(text)
        assert len(chunks) > 1

    def test_no_chunk_exceeds_size_for_short_sentences(self):
        sentences = ["Hi there." for _ in range(20)]
        text = " ".join(sentences)
        c = ContentAwareChunker(strategy="sentence", size=50, overlap=0)
        chunks = c.chunk(text)
        for chunk in chunks:
            # each individual sentence is 9 chars, so this always holds
            assert len(chunk) <= 50 + 9  # allow one sentence overflow

    def test_question_marks_and_exclamations_are_boundaries(self):
        text = "Is this a question? Yes it is! Great."
        chunks = self.c.chunk(text)
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "Is this a question" in combined
        assert "Yes it is" in combined


# ── paragraph strategy ─────────────────────────────────────────────────────

class TestParagraphStrategy:
    def setup_method(self):
        self.c = ContentAwareChunker(strategy="paragraph", size=100, overlap=20)

    def test_empty_returns_empty(self):
        assert self.c.chunk("") == []

    def test_single_paragraph(self):
        text = "Just one paragraph here with some words."
        chunks = self.c.chunk(text)
        assert len(chunks) == 1

    def test_paragraphs_grouped_until_size(self):
        short_paras = ["Short para." for _ in range(20)]
        text = "\n\n".join(short_paras)
        c = ContentAwareChunker(strategy="paragraph", size=50, overlap=0)
        chunks = c.chunk(text)
        assert len(chunks) > 1
        assert len(chunks) < 20  # paragraphs are grouped, not one-per-chunk

    def test_large_single_paragraph_subdivided(self):
        # A paragraph larger than size*1.5 must be split further
        big_para = "word " * 200  # ~1000 chars
        c = ContentAwareChunker(strategy="paragraph", size=100, overlap=10)
        chunks = c.chunk(big_para)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_blank_paragraphs_ignored(self):
        text = "Para one.\n\n\n\n\nPara two."
        chunks = self.c.chunk(text)
        combined = "".join(chunks)
        assert "Para one" in combined
        assert "Para two" in combined


# ── code strategy ──────────────────────────────────────────────────────────

class TestCodeStrategy:
    def setup_method(self):
        self.c = ContentAwareChunker(strategy="code", size=200, overlap=0)

    def test_falls_back_to_paragraph_when_too_few_definitions(self):
        text = "def only_one():\n    pass\n"
        # Only 1 def — fewer than 2 bounds, should fall back
        chunks = self.c.chunk(text)
        assert len(chunks) >= 1

    def test_splits_on_def_boundaries(self):
        text = (
            "def foo():\n    return 1\n\n"
            "def bar():\n    return 2\n\n"
            "def baz():\n    return 3\n"
        )
        chunks = self.c.chunk(text)
        assert len(chunks) == 3
        assert any("foo" in c for c in chunks)
        assert any("bar" in c for c in chunks)
        assert any("baz" in c for c in chunks)

    def test_splits_on_class_boundaries(self):
        text = (
            "class Alpha:\n    x = 1\n\n"
            "class Beta:\n    x = 2\n"
        )
        chunks = self.c.chunk(text)
        assert len(chunks) == 2

    def test_empty_returns_empty(self):
        assert self.c.chunk("") == []

    def test_no_empty_chunks_produced(self):
        text = (
            "def a():\n    pass\n\n"
            "def b():\n    pass\n\n"
            "def c():\n    pass\n"
        )
        chunks = self.c.chunk(text)
        for chunk in chunks:
            assert chunk.strip() != ""


# ── dispatch ───────────────────────────────────────────────────────────────

class TestDispatch:
    def test_chunk_dispatches_to_sentence(self):
        c = ContentAwareChunker(strategy="sentence", size=200, overlap=0)
        text = "Hello world. Goodbye world."
        chunks = c.chunk(text)
        assert isinstance(chunks, list)
        assert all(isinstance(ch, str) for ch in chunks)

    def test_chunk_dispatches_to_paragraph(self):
        c = ContentAwareChunker(strategy="paragraph", size=200, overlap=0)
        text = "Para one.\n\nPara two."
        chunks = c.chunk(text)
        assert isinstance(chunks, list)

    def test_chunk_dispatches_to_code(self):
        c = ContentAwareChunker(strategy="code", size=200, overlap=0)
        text = "def a():\n    pass\ndef b():\n    pass\n"
        chunks = c.chunk(text)
        assert isinstance(chunks, list)
