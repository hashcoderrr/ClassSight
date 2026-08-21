# Models Directory

This directory stores deep learning model weights used by ClassSight AI.

## Overview

- YOLO model weights are auto-downloaded on first run if not already present.
- Default model: `yolo11n.pt` (~6MB, YOLO11 Nano).
- Model weight files (`*.pt`, `*.onnx`, `*.engine`) are gitignored.
- To use a different model, change `MODEL_NAME` in `config.py` or specify via the `--model` CLI argument (or `CLASSSIGHT_MODEL` environment variable).
