#!/usr/bin/env bash
set -euo pipefail

MODEL_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.pl.300.bin.gz"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME="cc.pl.300.bin.gz"
MODEL_NAME="cc.pl.300.bin"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading FastText Polish model to $DEST_DIR..."

if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$ARCHIVE_NAME" "$MODEL_URL"
elif command -v wget >/dev/null 2>&1; then
    wget -O "$ARCHIVE_NAME" "$MODEL_URL"
else
    echo "Error: curl or wget is required to download the model." >&2
    exit 1
fi

echo "Download completed. Extracting..."

gunzip -f "$ARCHIVE_NAME"

echo "Model extracted to: $DEST_DIR/$MODEL_NAME"

echo "Done. You can now use the model at $DEST_DIR/$MODEL_NAME"
