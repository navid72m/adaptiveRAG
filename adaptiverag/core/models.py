from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List


@dataclass
class KBProfile:
    doc_count:      int = 0
    avg_doc_length: int = 0
    domain:         str = "general"
    structure_type: str = "mixed"
    complexity:     str = "moderate"
    has_code:       bool = False
    has_tables:     bool = False
    has_lists:      bool = False
    languages:      List[str] = field(default_factory=lambda: ["english"])

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"domain={self.domain}, structure={self.structure_type}, "
            f"complexity={self.complexity}, docs={self.doc_count}, "
            f"avg_len={self.avg_doc_length}ch, code={self.has_code}"
        )


@dataclass
class PipelineConfig:
    chunk_strategy: str   = "paragraph"
    chunk_size:     int   = 512
    chunk_overlap:  int   = 64
    top_k:          int   = 5
    temperature:    float = 0.5
    rerank_enabled: bool  = False

    def to_dict(self) -> Dict:
        return asdict(self)
