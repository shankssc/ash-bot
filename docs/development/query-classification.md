# Query Classification Guide

## Overview

The Query Classifier is the **routing brain** of AniRAG's RAG engine (Phase 2). It analyzes user queries to determine intent and routes them to specialized retrieval/generation strategies, enabling:

| Intent             | Retrieval Strategy                                | Generation Strategy                         | Example Queries                                                 |
| ------------------ | ------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------------- |
| **Factual**        | Strict metadata filtering + citation requirements | Precise, sourced answers                    | "Who directed Cowboy Bebop?", "What is Attack on Titan's plot?" |
| **Recommendation** | Genre/era similarity + diversity ranking          | Personalized suggestions with reasoning     | "Anime like Attack on Titan", "What should I watch next?"       |
| **Comparison**     | Dual-entity retrieval + contrastive synthesis     | Structured comparison with pros/cons        | "Naruto vs One Piece", "Compare Death Note and Code Geass"      |
| **Creative**       | Relaxed retrieval (broader context)               | LLM creativity mode + imaginative responses | "Write a haiku about Goku", "Roleplay as Spike Spiegel"         |
| **Meta**           | System capability lookup                          | Help/documentation responses                | "What can you do?", "How do you work?"                          |

```mermaid
flowchart TD
    Q[User Query] --> C[Query Classifier]
    C -->|Intent + Confidence| O[Orchestrator]

    subgraph O [Orchestrator]
        O -->|factual| F[Factual Handler<br>Strict filtering + citations]
        O -->|recommendation| R[Recommendation Handler<br>Similarity + diversity]
        O -->|comparison| P[Comparison Handler<br>Dual-entity retrieval]
        O -->|creative| V[Creative Handler<br>Relaxed context]
        O -->|meta| M[Meta Handler<br>System capabilities]
    end

    F --> G[Generation Engine]
    R --> G
    P --> G
    V --> G
    M --> G
    G --> A[Final Answer]
```
