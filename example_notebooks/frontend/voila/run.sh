#!/usr/bin/env bash
# set -euo pipefail

PROJECT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
voila --port 8787 --no-browser --Voila.ip 0.0.0.0 voila_demo.ipynb 
