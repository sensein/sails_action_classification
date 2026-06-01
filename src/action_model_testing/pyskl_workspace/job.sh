#!/bin/bash
#SBATCH --job-name=pyskl_train
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-17          # 9 models x 2 datasets = 18 jobs (0-17)
#SBATCH --output=/home/aparnabg/orcd/pool/pyskl_workspace/train_logs/slurm_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/pool/pyskl_workspace/train_logs/slurm_%A_%a.err

set -u

# ---- Paths ----
WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
LOG_DIR=${WORKSPACE}/train_logs
ENV_PATH=${WORKSPACE}/envs/pyskl

mkdir -p ${LOG_DIR}

# ---- Environment ----
module purge
module load miniforge
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6

echo "=========================================="
echo "  SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "  Host               : $(hostname)"
echo "  Date               : $(date)"
echo "  Python             : $(which python)"
echo "=========================================="
nvidia-smi | head -10
echo "=========================================="

cd ${PYSKL_ROOT}

# ---- Define all 18 jobs ----
# Format: "config_path"
JOBS=(
    # RMM dataset (4 classes) — indices 0-8
    "configs/custom/stgcnpp_rmm/j.py"
    "configs/custom/stgcnpp_rmm/b.py"
    "configs/custom/stgcnpp_rmm/jm.py"
    "configs/custom/stgcnpp_rmm/bm.py"
    "configs/custom/ctrgcn_rmm/j.py"
    "configs/custom/ctrgcn_rmm/b.py"
    "configs/custom/ctrgcn_rmm/jm.py"
    "configs/custom/ctrgcn_rmm/bm.py"
    "configs/custom/posec3d_rmm/joint.py"
    # LOCO dataset (5 classes) — indices 9-17
    "configs/custom/stgcnpp_loco/j.py"
    "configs/custom/stgcnpp_loco/b.py"
    "configs/custom/stgcnpp_loco/jm.py"
    "configs/custom/stgcnpp_loco/bm.py"
    "configs/custom/ctrgcn_loco/j.py"
    "configs/custom/ctrgcn_loco/b.py"
    "configs/custom/ctrgcn_loco/jm.py"
    "configs/custom/ctrgcn_loco/bm.py"
    "configs/custom/posec3d_loco/joint.py"
)

CONFIG=${JOBS[$SLURM_ARRAY_TASK_ID]}
echo "Running config: ${CONFIG}"

# ---- Train ----
bash tools/dist_train.sh ${CONFIG} 1 --validate --test-last --test-best

echo "Done: ${CONFIG}"
echo "Finished at: $(date)"