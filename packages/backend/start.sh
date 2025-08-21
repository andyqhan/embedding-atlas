#!/bin/bash

export OMP_NUM_THREADS=1
export UMAP_DISABLE_MULTIPROCESSING=1
export NUMBA_NUM_THREADS=1

# uv run embedding-atlas spawn99/wine-reviews --text description --split train --static ../viewer/dist "$@"
uv run embedding-atlas ~/Documents/personal-coding/journal-ai-analysis/journal_excerpts.parquet --text excerpt --static ../viewer/dist "$@"
