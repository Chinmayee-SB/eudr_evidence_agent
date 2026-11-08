"""Rerank search candidates with a cross-encoder: a model that reads the question and a passage together.

Keyword and vector search each score a passage on its own, so the merge can only guess at the order. A
cross-encoder sees both texts at once and scores how well the passage answers the question, which puts the
right passage near the top (measured in eval/run_rag_quality.py: MRR, nDCG).

Model: BAAI bge-reranker-base, the quantized ONNX build (Xenova/bge-reranker-base, about 270 MB, downloaded once
to the Hugging Face cache). It runs on onnxruntime and tokenizers, which Chroma already installs, so no new
package is needed. RAG_RERANKER=none switches it off. If the model cannot load, search goes on without it and
the trace says so.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

MODEL = os.getenv("RAG_RERANKER", "Xenova/bge-reranker-base")
MAX_TOKENS = 512   # the model's limit; question and passage together
BATCH = 16


@lru_cache(maxsize=1)
def _model() -> tuple[Any, Any]:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, "tokenizer.json"))
    tokenizer.enable_truncation(MAX_TOKENS)
    tokenizer.enable_padding()
    session = ort.InferenceSession(hf_hub_download(MODEL, "onnx/model_quantized.onnx"),
                                   providers=["CPUExecutionProvider"])
    return tokenizer, session


def enabled() -> bool:
    return MODEL.lower() not in {"", "none", "off"}


def scores(question: str, passages: list[str]) -> list[float]:
    """One relevance score per passage, higher is better. Raises if the model cannot be loaded."""
    import numpy as np

    tokenizer, session = _model()
    out: list[float] = []
    for start in range(0, len(passages), BATCH):
        batch = tokenizer.encode_batch([(question, p) for p in passages[start:start + BATCH]])
        logits = session.run(["logits"], {
            "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64)})[0]
        out += [float(x) for x in logits[:, 0]]
    return out
