#!/usr/bin/env bash
# Restore every dataset the reported results use, then check it against data/USED.csv.
#   release  (default, recommended) unpack the release archives: the exact input/ + labels/ + feature cache used for
#            the reported numbers. Split order depends on edgelist LINE ORDER, so this is the only exact route.
#   rebuild  re-download from the original sources with the builders in data/USED.csv. Hosts change and re-downloads
#            can renumber nodes, so every file is hash-checked and any mismatch is reported, never silently accepted.
set -euo pipefail
MODE="${1:-release}"
REL="${RELEASE_DIR:-release}"                      # folder holding the downloaded release assets
cd "$(dirname "$0")/.."

if [ "$MODE" = "release" ]; then
    (cd "$REL" && sha256sum -c SHA256SUMS)
    tar -xzf "$REL/virgo-input-labels.tar.gz"
    mkdir -p output && tar -xzf "$REL/virgo-feature-cache.tar.gz" -C output
elif [ "$MODE" = "rebuild" ]; then
    tail -n +2 data/USED.csv | cut -d, -f2 | sort -u | grep '^python ' | while read -r cmd; do
        echo ">> $cmd"; $cmd
    done
    echo "NOTE: cora, citeseer_linqs, enzymes, proteins need the Identity2Vec author's input.zip edgelists (see AGENT.md)."
else
    echo "usage: $0 [release|rebuild]"; exit 1
fi

python verify/compare.py --step hashes --kinds dataset feature_cache final_result --report verify/report_data.csv
