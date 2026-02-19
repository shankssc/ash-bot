````markdown
# Retrieval Architecture: Hybrid Search Flow

## Overview

AniRAG implements a **hybrid retrieval architecture** combining dense vector search (semantic similarity) with sparse BM25 search (keyword matching). Results are fused using Reciprocal Rank Fusion (RRF) to leverage strengths of both approaches.

```mermaid
flowchart TB
    Q[User Query] --> C[Query Classifier]
    C -->|Intent| R[Retrieval Engine]

    subgraph R [Retrieval Engine]
        Q --> D[Dense Search<br>Qdrant Vector DB]
        Q --> S[Sparse Search<br>BM25 on Synopsis Text]
        D --> F[Reciprocal Rank Fusion<br>k=60]
        S --> F
        F --> T[Top-K Results<br>with Metadata]
    end

    T --> G[Generation Engine]
    G --> A[Final Answer]

    subgraph Caching [Optional Optimization]
        Q -.->|Semantic Cache Check| SC{Cache Hit?}
        SC -->|Yes| A
        SC -->|No| R
    end
```
````
