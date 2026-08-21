# Data Directory

This directory contains input datasets, test images, and test videos for ClassSight AI.

## Directory Structure

- `images/`: Place test and sample images here (e.g. `test.jpg`).
- `videos/`: Place test and classroom video files here (e.g. `test.mp4`).
- `ouc-cge/`: Default directory for the OUC-CGE dataset (or configure a custom location).

## Usage Guidelines

- Place test images in `data/images/`.
- Place test videos in `data/videos/`.
- For the OUC-CGE dataset: set the path in `config.py` or via the `CLASSSIGHT_OUC_CGE_ROOT` environment variable.
- **Do not commit large dataset files to git.** Large media and dataset folders are ignored via `.gitignore`.
