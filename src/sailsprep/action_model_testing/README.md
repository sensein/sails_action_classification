# pyskl custom pipeline

This repo does **not** contain the [pyskl](https://github.com/kennymckormick/pyskl)
codebase itself. It's a set of custom scripts (data conversion, config
generation, SLURM job scripts, and late-fusion training) that must be dropped
into an actual clone of the upstream `pyskl` repo to run.

## Setup

1. Clone the upstream repo and install it (see its README for full
   environment setup, e.g. mmcv, torch, etc.):

   ```bash
   git clone https://github.com/kennymckormick/pyskl.git
   cd pyskl
   pip install -e .
   ```

2. Copy the contents of this repo into the root of that clone:

   ```bash
   cp -r scripts fusion jobs /path/to/pyskl/
   ```

   (`tools/dist_train.sh`, `tools/dist_test.sh`, and the `pyskl` Python
   package used by `extract_logits.py` all come from the upstream clone —
   they are not included here.)

## Pipeline / run order

All commands below are run **from the root of the upstream `pyskl` clone**
(i.e. `cd /path/to/pyskl` first), after copying these files in — not from
inside `scripts/` or `fusion/`. The scripts read/write paths like
`configs/custom/...` and `work_dirs/...` relative to your current directory,
not relative to the script's own location.

1. **Convert raw pose JSONs to pyskl's `.pkl` annotation format:**

   ```bash
   python scripts/pyskl_dataset.py --task rmm  --out /path/to/rmm_pyskl.pkl
   python scripts/pyskl_dataset.py --task loco --out /path/to/loco_pyskl.pkl
   ```

2. **Generate model configs** (writes into `configs/custom/<model>_<dataset>/`):

   ```bash
   python scripts/generate_stgcnpp_configs.py
   python scripts/generate_ctrgcn_configs.py
   python scripts/generate_posec3d_configs.py
   ```

3. **Train** all model/feature combinations across 3 seeds via SLURM
   (edit `WORKSPACE` in the script, or set it as an env var, to match your
   cluster paths):

   ```bash
   bash jobs/pyskl/pyskl_job.sh
   ```

4. **Test** the best checkpoint from each run:

   ```bash
   bash jobs/pyskl/test_job.sh
   ```

5. **Extract per-model logits** on val + test splits, for fusion:

   ```bash
   python fusion/extract_logits.py --dataset rmm  --split val
   python fusion/extract_logits.py --dataset rmm  --split test
   python fusion/extract_logits.py --dataset loco --split val
   python fusion/extract_logits.py --dataset loco --split test
   ```

6. **Train the fusion MLP** over the concatenated logits and evaluate:

   ```bash
   python fusion/train_mlp_fusion.py --dataset rmm
   python fusion/train_mlp_fusion.py --dataset loco
   ```


