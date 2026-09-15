#!/bin/bash
# Build-time install step for CML Model builds.
# predict.py imports `portfolio_optimization`, which is not installed in a
# fresh build container by default — install it (and its deps) here first.
set -e
pip install -r requirements.txt
pip install -e .
