| Variant (section chunking) | Retrieval | Hit@5 | MRR@10 (95% CI) | p vs default | ms/query |
|---|---|---|---|---|---|
| BGE-small, 180 words, heading prefix (default) | bm25 | 0.721 | 0.506 (0.41–0.61) | — | 1 |
| BGE-small, 180 words, heading prefix (default) | dense | 0.738 | 0.596 (0.49–0.69) | — | 6 |
| BGE-small, 180 words, heading prefix (default) | hybrid | 0.787 | 0.574 (0.48–0.68) | — | 8 |
| BGE-small, 180 words, heading prefix (default) | hybrid_rerank | 0.803 | 0.690 (0.59–0.79) | — | 159 |
| MiniLM-L6 embeddings instead of BGE-small | bm25 | 0.721 | 0.506 (0.41–0.61) | 1.000 | 1 |
| MiniLM-L6 embeddings instead of BGE-small | dense | 0.656 | 0.417 (0.32–0.51) | 0.001 | 27 |
| MiniLM-L6 embeddings instead of BGE-small | hybrid | 0.803 | 0.531 (0.44–0.62) | 0.304 | 6 |
| MiniLM-L6 embeddings instead of BGE-small | hybrid_rerank | 0.820 | 0.711 (0.62–0.80) | 0.110 | 154 |
| 100-word chunks | bm25 | 0.607 | 0.500 (0.39–0.61) | 0.869 | 2 |
| 100-word chunks | dense | 0.721 | 0.553 (0.45–0.65) | 0.281 | 6 |
| 100-word chunks | hybrid | 0.770 | 0.541 (0.44–0.64) | 0.421 | 9 |
| 100-word chunks | hybrid_rerank | 0.820 | 0.696 (0.60–0.80) | 0.847 | 115 |
| 300-word chunks | bm25 | 0.721 | 0.541 (0.44–0.64) | 0.239 | 1 |
| 300-word chunks | dense | 0.770 | 0.635 (0.53–0.73) | 0.198 | 6 |
| 300-word chunks | hybrid | 0.787 | 0.616 (0.51–0.71) | 0.321 | 8 |
| 300-word chunks | hybrid_rerank | 0.820 | 0.714 (0.62–0.81) | 0.379 | 174 |
| no heading prefix | bm25 | 0.705 | 0.491 (0.39–0.59) | 0.605 | 1 |
| no heading prefix | dense | 0.787 | 0.533 (0.44–0.62) | 0.083 | 6 |
| no heading prefix | hybrid | 0.754 | 0.563 (0.46–0.66) | 0.763 | 8 |
| no heading prefix | hybrid_rerank | 0.820 | 0.639 (0.54–0.74) | 0.187 | 160 |
