from __future__ import annotations

import json
import re
import statistics
from typing import Dict, List

from ..core.config import CHUNK_SIZE_OPTIONS
from ..core.models import KBProfile, PipelineConfig
from ..core.runtime import RT

try:
    from sentence_transformers import CrossEncoder  # noqa: F401
    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


def profile_documents(documents: List[Dict]) -> KBProfile:
    p = KBProfile()
    if not documents:
        return p

    lengths          = [len(d["content"]) for d in documents]
    p.doc_count      = len(documents)
    p.avg_doc_length = int(statistics.mean(lengths))
    all_text         = " ".join(d["content"] for d in documents)

    code_pats = [
        r'\bdef \w+\(', r'\bclass \w+[:\(]', r'\bimport \w+',
        r'\bfunction\s+\w+\s*\(', r'```[\w]*\n',
    ]
    p.has_code   = sum(1 for pat in code_pats if re.search(pat, all_text)) >= 2
    p.has_tables = bool(re.search(r'\|.+\|.+\|', all_text, re.MULTILINE))
    p.has_lists  = bool(re.search(r'^\s*[-•*]\s+\w+', all_text, re.MULTILINE))

    sample = "\n\n---\n\n".join(
        f"[{d['filename']}]\n{d['content'][:1500]}" for d in documents[:5]
    )
    prompt = f"""Classify these documents. Return ONLY JSON:
{{
  "domain": "<medical|legal|technical|financial|code|scientific|general>",
  "structure_type": "<narrative|qa|reference|code|mixed>",
  "complexity": "<simple|moderate|complex>"
}}

SAMPLES:
{sample}"""
    try:
        resp = RT.llm.invoke(prompt).content
        m    = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            data             = json.loads(m.group())
            p.domain         = data.get("domain", "general")
            p.structure_type = data.get("structure_type", "mixed")
            p.complexity     = data.get("complexity", "moderate")
    except Exception as e:
        print(f"  ✗   KB classification failed: {e}")
    return p


def initial_config(profile: KBProfile) -> PipelineConfig:
    cfg = PipelineConfig()

    if profile.avg_doc_length < 500:
        cfg.chunk_size, cfg.chunk_overlap, cfg.chunk_strategy = 200, 20, "sentence"
    elif profile.avg_doc_length < 2000:
        cfg.chunk_size, cfg.chunk_overlap = 400, 40
    elif profile.avg_doc_length < 8000:
        cfg.chunk_size, cfg.chunk_overlap = 700, 90
    else:
        cfg.chunk_size, cfg.chunk_overlap = 1024, 128

    if profile.has_code:
        cfg.chunk_strategy = "code"
        cfg.chunk_size     = 800
        cfg.top_k          = 7

    if profile.domain in ("medical", "legal", "scientific"):
        cfg.top_k          = 8
        cfg.temperature    = 0.3
        cfg.rerank_enabled = _CROSS_ENCODER_AVAILABLE
    elif profile.domain == "financial":
        cfg.temperature = 0.4

    if profile.complexity == "complex":
        cfg.top_k       = min(cfg.top_k + 2, 12)
        cfg.temperature = max(cfg.temperature - 0.1, 0.1)

    return cfg
