#!/bin/sh
# First start: download the corpus and build the default index into the mounted volumes.
set -e
if [ ! -f /app/models/index/chunks_section.jsonl ]; then
  echo "No index yet: downloading the corpus and building it (a few minutes on CPU)..."
  rag-download
  rag-index
fi
exec "$@"
