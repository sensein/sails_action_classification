# analysis

Behavior-specific statistical analysis scripts. Each script extracts
kinematic features from pose keypoints for one behavior, then runs a battery
of statistical tests comparing the ASD and non-ASD groups (mixed-effects
models, GEE, cluster-robust regression, bootstrap effect sizes, permutation
tests, ICC reliability, and leave-one-subject-out classification).

## Layout

```
analysis/
  common/
    banners.py                     hr_v1/hr_v2 — printed section headers
    bayes.py                       Bayesian helper: Savage-Dickey Bayes factor, standardisation
    consensus.py                   make_consensus — flags features significant across multiple methods
    cross_validation.py            run_loso_child — leave-one-subject-out classification with permutation p-value
    effect_size.py                 cohen_d_v1/v2/v3 — Cohen's d variants
    icc.py                         compute_icc — intraclass correlation for repeated sessions
    keypoints.py                   get_kp, assign_age_band, torso_length, get_scale
    mixed_models.py                _use_random_slope — decides fixed vs random-slope LME structure
    parsing.py                     extract_pid, extract_session, timestamp parsing
    signal_processing.py           Butterworth filtering, 2D joint angles, SPARC smoothness, spectral features
    significance.py                FDR correction and significance-bar annotation for plots
    crawling_running_stats.py      bootstrap CI, permutation test, LME (Kenward-Roger), cluster-robust (CR2) — shared by crawling/running
    crusing_walking_stats.py       bootstrap CI helper shared by cruising/walking
    handflapping_spinning_stats.py bootstrap CI helper shared by handflapping/spinning
  crawling/crawling.py
  crusing/crusing.py
  running/running.py
  walking/walking.py
  handflapping/handflapping.py
  jumping/jumping.py
  rocking/rocking.py
  spinning/spinning.py
  loco_combined/loco_combined.py   combined analysis across all locomotion behaviors
  rmm_combined/rmm_combined.py     combined analysis across all repetitive-motor-movement behaviors
```

Each behavior script is a single standalone file (~1,400-2,000 lines) with no
CLI arguments — input CSV paths and the output directory are constants near
the top of the file, and the whole script runs top to bottom as one pipeline
(load data -> extract features -> run stats -> generate figures -> print
summary).

## Usage

Update the path constants at the top of the script for your environment, then
run it as a module (imports are `sailsprep.analysis.common.*`, which requires
the package to be installed, e.g. via `poetry install`):

```bash
python -m sailsprep.analysis.walking.walking
python -m sailsprep.analysis.crawling.crawling
python -m sailsprep.analysis.running.running
python -m sailsprep.analysis.crusing.crusing
python -m sailsprep.analysis.handflapping.handflapping
python -m sailsprep.analysis.jumping.jumping
python -m sailsprep.analysis.rocking.rocking
python -m sailsprep.analysis.spinning.spinning
python -m sailsprep.analysis.loco_combined.loco_combined
python -m sailsprep.analysis.rmm_combined.rmm_combined
```

Running the script file directly (`python src/sailsprep/analysis/walking/walking.py`)
also works once the package is installed, since the scripts only import from
`sailsprep.analysis.*`, not from files relative to their own folder.

`jobs/analysis/analysis_job.sh` runs the ten behavior scripts as a SLURM job
array.

## Run order

Most behaviors are independent, with one exception: `crusing/crusing.py`
reads `walking/v3/child_level_features.csv`, the child-level feature CSV
written by `walking.py`, so `walking.py` must be run first.

## Inputs

Scripts read one or more of:
- `MAIN_CSV` / `SPLITS_CSV` — the master split CSV with video, pose, and
  label paths.
- `RMM_CSV` — an additional clip-to-annotation matching CSV used by the RMM
  behaviors.
- Per-frame pose keypoint data referenced from the split CSV, read via
  `common/keypoints.py`.

## Outputs

Each script writes into its own `OUTPUT_DIR`/`BASE_DIR`: a
`child_level_features.csv` of extracted per-child kinematic features,
statistical result CSVs (LME, GEE, permutation, bootstrap, ICC), a
`figures/` subdirectory of plots, and (when PyMC is available) Bayesian
sensitivity analysis output.

## Optional dependencies

Several scripts try to import `rpy2` (for R-based Kenward-Roger corrected
mixed models), `pymc`/`arviz` (Bayesian models), and `wildboottest` (cluster
wild bootstrap / CR2 standard errors) inside a `try/except`, and print which
of these are available at startup. If a package is missing, the
corresponding section of the analysis is skipped rather than the script
failing — statsmodels-based mixed models and bootstrap/permutation tests
still run either way. `rpy2`, `pymc`, and `arviz` are covered by the
`stats-analysis` Poetry group (`poetry install --with dev,stats-analysis`);
`wildboottest` is not in `pyproject.toml` and needs a separate `pip install
wildboottest` if you want CR2 standard errors.
