# Minimal Ingestion Pipeline Output

Generated on: 1.0.0
Duration: 3.73 seconds
Anime processed: 20
Chunks created: 76
Embedding dimension: 384

## Files
- `anime_metadata.json`: Basic anime information
- `chunks.json`: Text chunks with metadata
- `embeddings.npy`: Embedding vectors (numpy array)
- `embedding_metadata.json`: Per-embedding metadata
- `MANIFEST.json`: Pipeline execution metadata

## Usage
```python
import numpy as np
embeddings = np.load('embeddings.npy')
