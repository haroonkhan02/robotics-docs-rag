#!/bin/sh
# Ablations on top of the main grid (`rag-index --strategy all && rag-eval-retrieval`).
# Each writes reports/retrieval/results_<tag>.json. About 5 minutes on an Apple M5.
set -e
MINILM="embedding.model=sentence-transformers/all-MiniLM-L6-v2"

# 1. Embedding model: MiniLM (no query instruction) instead of BGE-small, section chunks
rag-index --strategy section --set "$MINILM" --set embedding.query_prefix="''" \
  --stats-out reports/index_stats.json
rag-eval-retrieval --strategies section --set "$MINILM" --set embedding.query_prefix="''" \
  --tag emb_minilm

# 2. Chunk size: 100 and 300 words instead of 180 (section chunks, BGE-small)
for W in 100 300; do
  rag-index --strategy section --set chunking.max_words=$W --set index_dir=models/index_w$W
  rag-eval-retrieval --strategies section --set chunking.max_words=$W \
    --set index_dir=models/index_w$W --tag section_w$W
done

# 3. Heading prefix: section boundaries kept, "Doc > Section" prefix removed
rag-index --strategy section --set chunking.section_header=false --set index_dir=models/index_nohdr
rag-eval-retrieval --strategies section --set chunking.section_header=false \
  --set index_dir=models/index_nohdr --tag section_noheader
