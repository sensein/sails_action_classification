#!/bin/bash
#SBATCH --job-name=pyskl_inf
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/aparnabg/orcd/pool/pyskl_workspace/logs/pyskl_inf_%A_%a.out
#SBATCH --error=/home/aparnabg/orcd/pool/pyskl_workspace/logs/pyskl_inf_%A_%a.err

TASK=$1

WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
LOG_DIR=${WORKSPACE}/train_logs
ENV_PATH=${WORKSPACE}/envs/pyskl

mkdir -p ${LOG_DIR}

# ---- Environment ----
module purge
module load miniforge
conda deactivate

source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6


SEEDS=(42 456 123)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Task: ${TASK}  Seed: ${SEED}  $(date)"
python /home/aparnabg/orcd/pool/pyskl_workspace/pyskl_full_video_inference.py \
    --task ${TASK} --seed ${SEED}
echo "Done: $(date)"