#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-$REPO_ROOT/experimental_lab/registries/external_repos.tsv}"
EXTERNAL_DIR="$REPO_ROOT/experimental_lab/external"

mkdir -p "$EXTERNAL_DIR"

echo "Using manifest: $MANIFEST"
echo "External directory: $EXTERNAL_DIR"
echo

while IFS=$'\t' read -r paper_id folder url ref notes; do
    # Skip empty lines and comments
    if [[ -z "${paper_id:-}" ]] || [[ "$paper_id" == \#* ]]; then
        continue
    fi

    if [[ -z "${folder:-}" ]] || [[ -z "${url:-}" ]]; then
        echo "[skip] Invalid row for paper_id=$paper_id"
        continue
    fi

    if [[ "$url" == "UNAVAILABLE" ]] || [[ "$url" == "TODO_URL" ]]; then
        echo "[skip] Paper $paper_id ($folder): $url"
        continue
    fi

    dest="$EXTERNAL_DIR/$folder"

    if [[ -d "$dest/.git" ]]; then
        echo "[exists] Paper $paper_id: $folder"
        echo "         $(git -C "$dest" rev-parse --short HEAD 2>/dev/null || true)"
        continue
    fi

    if [[ -e "$dest" ]]; then
        echo "[skip] Paper $paper_id: $dest exists but is not a Git repo"
        continue
    fi

    echo "[clone] Paper $paper_id: $url -> $dest"
    git clone "$url" "$dest"

    if [[ -n "${ref:-}" ]] && [[ "$ref" != "-" ]]; then
        echo "[checkout] $folder -> $ref"
        git -C "$dest" checkout "$ref"
    fi

    echo "[done] $folder @ $(git -C "$dest" rev-parse --short HEAD)"
    echo
done < "$MANIFEST"

echo "External repository setup finished."
