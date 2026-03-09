# AniRAG 🎌

Production-grade anime intelligence microservice built on **100% free infrastructure**.

[![CI](https://github.com/shankssc/ash-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/shankssc/ash-bot/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## ✨ Key Differentiators

- **Hybrid RAG**: Dense vectors + sparse BM25 with reciprocal rank fusion
- **Intent-aware routing**: Specialized prompts for factual/recommendation/comparison queries
- **Semantic caching**: 95%+ similarity matching to reduce latency/costs
- **Production patterns**: Rate limiting, circuit breakers, structured logging
- **Resource-constrained engineering**: Maximizes free tiers (Qdrant 1GB, Upstash 10MB, etc.)

## 📊 Project Status

| Phase | Component                  | Status      | Target Date |
| ----- | -------------------------- | ----------- | ----------- |
| ✅ 0  | Project Foundations        | Complete    | Week 1      |
| ✅ 1A | Minimal Ingestion Pipeline | Complete    | Week 2      |
| ✅ 1B | Qdrant Integration         | Complete    | Week 3      |
| ✅ 2A | Query Classifier           | Complete    | Week 3      |
| ✅ 2B | Hybrid Search Engine       | Complete    | Week 4      |
| ⏳ 2C | Semantic Caching           | Not Started | Week 4-5    |
| ⏳ 3  | API Layer                  | Not Started | Week 5      |
| ⏳ 4  | Evaluation                 | Not Started | Week 6      |
| ✅ 5  | Documentation              | In Progress | Week 6-7    |

## 🚀 Quick Start

```bash
# 1. Clone repo and install dependencies
git clone https://github.com/shankssc/ash-bot.git
cd ash-bot
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# 2. Setup environment (copy template and fill values)
cp .env.example .env
# Edit .env with your API keys:
#   - QDRANT_URL / QDRANT_API_KEY (from https://cloud.qdrant.io)
#   - REDIS_URL / REDIS_TOKEN (from https://console.upstash.com)
#   - GROQ_API_KEY (from https://console.groq.com/keys)

# 3. Ingest anime data into Qdrant Cloud
python scripts/run_minimal_pipeline.py --max-anime 20 --to-qdrant

# 4. Run health check
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/api/v1/health

# 5. Test hybrid search manually
python scripts/test_sparse_search.py "magic adventure elf" --top-k 3
```
