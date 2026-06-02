# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview
Academic group project for **CI-0131 Diseño de Experimentos** (UCR, Escuela de Ciencias de la Computación). The experiment is a **2×3 full factorial** comparing two human-pose-estimation algorithms on single-person static images from **COCO val2017**:
- **Factor A — algorithm** (2 levels): MediaPipe BlazePose Full vs. MoveNet Thunder v4.
- **Factor B — input resolution** (3 levels): high / medium / low.
- **Response variable:** `PCK@0.5` (Percentage of Correct Keypoints, threshold = 50% of the person's bounding-box diagonal; only keypoints with COCO visibility flag = 2). Continuous, 0–100%.

Core hypothesis: MoveNet (heatmap-based) degrades more under resolution loss than BlazePose (direct regression). Working language of all reports and deliverables is **Spanish**.

## Model preference
Default (Sonnet 4.6). Use `claude-opus-4-7` for statistical-design reasoning or report structuring.

## Architecture
- `scripts/split_indoor_outdoor_blocks.py` — classifies COCO images as indoor/outdoor/ambiguous. Emits `data/images_indoor.json`, `data/images_outdoor.json`, `data/images_ambiguous.json`, `data/blocks_summary.json`. Dataset-prep tooling, NOT the experiment runner.
- `scripts/download_annotations.py` — downloads COCO val2017 annotation zip and extracts the two needed JSONs to `Archive/annotations/`.
- `scripts/download_and_organize.py` — downloads only the 1 503 needed COCO images into `Archive/images/indoor/` and `Archive/images/outdoor/`.
- `experiment.py` — main runner. Loads GT from `Archive/annotations/`, selects 300 images from `data/`, runs BlazePose + MoveNet at 3 resolutions, prints PCK@0.5 per observation to stdout.
- `data/images_*.json` / `data/blocks_summary.json` — committed outputs of the split script (image_id + file_name pairs; counts).
- `docs/propuesta_final.docx` — full first-deliverable proposal (factors, levels, response, nuisance factors, randomization). Read this for experimental-design context.
- `docs/proyecto_enunciado.pdf` — assignment spec for the current "Avance" deliverable.
- `Archive/` — COCO data + annotations; **gitignored and not present locally**.

## Running the scripts (always from project root)
```bash
python scripts/download_annotations.py
python scripts/download_and_organize.py
python scripts/split_indoor_outdoor_blocks.py            # outputs to data/
python scripts/split_indoor_outdoor_blocks.py --include-non-keypoints
python experiment.py
```
No test suite, linter, or build step exists.

## Conventions
- **The graded analysis must be delivered as an R script + data file** (assignment requirement), even though current helper code is Python. EDA / statistical work belongs in R.
- Use the **effects model** (not the means model) when writing the factorial model and stating null/alternative hypotheses — explicitly required by the assignment.
- Randomization uses **seed = 42** (300 images assigned across the 6 treatments).
- Reports must be self-contained prose (no proposal template skeleton), Spanish, delivered as PDF.

## Active context
- Current deliverable: **"Avance"** (refined proposal + experimental-design definition + exploratory data analysis), 15% of course grade. **Due Monday 2026-06-08.**
- Required EDA: response-variable summary stats (mean, variance, sd, min/max, median, Q1/Q3), boxplot, histogram, time-order/randomization plot, per-factor and combined boxplots, interaction plots, plus a preliminary written interpretation.
- Known gotcha: experiment data has not yet been generated in-repo; the algorithm-runner that produces `PCK@0.5` measurements does not exist here yet.
