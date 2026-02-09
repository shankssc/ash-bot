# ADR 001: Embedding Model Selection

**Status**: Accepted  
**Date**: 2026-02-08  
**Decision Maker**: shankssc

## Context

Need embedding model that:

- Fits Qdrant Cloud free tier (1GB storage ≈ 150K vectors)
- Good semantic quality for anime knowledge
- CPU-friendly (Render free tier = 512MB RAM, no GPU)
- Fast inference for real-time queries

## Options Considered

| Model               | Dimensions | Quality             | Vectors in 1GB | CPU Speed | Verdict                                                      |
| ------------------- | ---------- | ------------------- | -------------- | --------- | ------------------------------------------------------------ |
| `all-MiniLM-L6-v2`  | 384        | Good (MTEB: 58.8)   | ~150K          | ⚡ Fast   | ✅ **Selected**                                              |
| `bge-small-en-v1.5` | 384        | Better (MTEB: 62.1) | ~150K          | Fast      | ❌ Requires sentence-transformers >=2.3 (compatibility risk) |
| `all-MiniLM-L12-v2` | 384        | Better (MTEB: 60.5) | ~150K          | 🐌 Slower | ❌ 2x slower than L6, not worth marginal gain                |
| `bge-large-en-v1.5` | 1024       | Best (MTEB: 64.3)   | ~55K           | 🐌 Slow   | ❌ Exceeds free tier capacity                                |

## Decision

**Select `sentence-transformers/all-MiniLM-L6-v2`** (384 dimensions)

## Consequences

✅ **Positive**

- Fits free tier constraints perfectly
- Fast CPU inference (<50ms per query on Render free tier)
- Proven quality for general knowledge tasks
- Mature library support (sentence-transformers 2.2+)

⚠️ **Negative**

- Slightly lower MTEB score than larger models (acceptable tradeoff)
- May struggle with highly nuanced anime comparisons (mitigated by hybrid BM25 search)

## Validation

Tested with 100 anime synopses:

- Semantic similarity for "space western" → Cowboy Bebop: 0.89 ✓
- "magical girl" → Sailor Moon: 0.92 ✓
- "isekai" → Re:Zero: 0.87 ✓
