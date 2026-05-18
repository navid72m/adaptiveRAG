from __future__ import annotations

import re
from typing import List


class ContentAwareChunker:
    def __init__(self, strategy: str = "paragraph", size: int = 512, overlap: int = 64):
        self.strategy = strategy
        self.size     = size
        self.overlap  = overlap

    def chunk(self, text: str) -> List[str]:
        return {
            "sentence":  self._sentence,
            "paragraph": self._paragraph,
            "code":      self._code,
        }.get(self.strategy, self._fixed)(text)

    def _fixed(self, text: str) -> List[str]:
        out, start = [], 0
        while start < len(text):
            out.append(text[start:start + self.size])
            start += self.size - self.overlap
        return [c for c in out if c.strip()]

    def _sentence(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text)
        out, cur = [], ""
        for s in sents:
            if cur and len(cur) + len(s) > self.size:
                out.append(cur.strip())
                cur = (cur[-self.overlap:] + " " + s) if self.overlap else s
            else:
                cur += (" " if cur else "") + s
        if cur.strip():
            out.append(cur.strip())
        return out

    def _paragraph(self, text: str) -> List[str]:
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        out, cur = [], ""
        for p in paras:
            if cur and len(cur) + len(p) > self.size:
                out.append(cur)
                cur = p
            else:
                cur += ("\n\n" if cur else "") + p
        if cur:
            out.append(cur)
        result = []
        for c in out:
            result.extend(self._fixed(c) if len(c) > self.size * 1.5 else [c])
        return result

    def _code(self, text: str) -> List[str]:
        bounds = list(re.finditer(
            r'^(?:def |class |function |public )', text, re.MULTILINE
        ))
        if len(bounds) < 2:
            return self._paragraph(text)
        chunks = []
        for i, m in enumerate(bounds):
            end   = bounds[i + 1].start() if i + 1 < len(bounds) else len(text)
            block = text[m.start():end].strip()
            chunks.extend(self._fixed(block) if len(block) > self.size * 2 else [block])
        return [c for c in chunks if c]
