#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/original_datasets
mkdir -p data/parquets
mkdir -p data/prepared
mkdir -p data/splits
mkdir -p experimental_lab/data

rm -rf experimental_lab/data/original_datasets
rm -rf experimental_lab/data/parquets

ln -s ../../data/original_datasets experimental_lab/data/original_datasets
ln -s ../../data/parquets experimental_lab/data/parquets

echo "Data symlinks created:"
ls -l experimental_lab/data
