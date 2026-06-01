#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-2

WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
ENV_PATH=${WORKSPACE}/envs/pyskl
mkdir -p ${WORKSPACE}/train_logs

MODEL=$1
TASK=$2

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge
module load miniforge
module load cuda/12.9.1
export PATH=/usr/bin:/bin:${ENV_PATH}/bin:$PATH
export CONDA_PREFIX=${ENV_PATH}
conda deactivate 
source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6


cd ${PYSKL_ROOT}

if [[ "$MODEL" == "posec3d" ]]; then
    CONFIG="configs/custom/posec3d_${TASK}_sw/joint.py"
elif [[ "$MODEL" == "ctrgcn" ]]; then
    CONFIG="configs/custom/ctrgcn_${TASK}_sw/jm.py"
else
    echo "ERROR: model must be posec3d or ctrgcn"; exit 1
fi

SEED_CONFIG=/tmp/${MODEL}_${TASK}_sw_s${SEED}.py
cp ${CONFIG} ${SEED_CONFIG}

# Patch work_dir for seed AND add resume_from if checkpoint exists
WORK_DIR="./work_dirs/${MODEL}_${TASK}_sw/$(basename ${CONFIG%.py})_s${SEED}"

python - << PYEOF
import re, os
with open('${SEED_CONFIG}', 'r') as f:
    content = f.read()
content = re.sub(r"work_dir = '(.*?)'", r"work_dir = '${WORK_DIR}'", content)

# Add resume_from if latest.pth exists
latest = '${WORK_DIR}/latest.pth'
if os.path.exists(latest):
    content += f"\nresume_from = '{latest}'\n"
    print(f"RESUMING from {latest}")
else:
    print("Starting fresh")

with open('${SEED_CONFIG}', 'w') as f:
    f.write(content)
PYEOF

echo "Model: ${MODEL}  Task: ${TASK}  Seed: ${SEED}  $(date)"
echo "Config: ${SEED_CONFIG}"
echo "Work dir: ${WORK_DIR}"

bash tools/dist_train.sh ${SEED_CONFIG} 1 \
    --validate --test-last --test-best --seed ${SEED}

echo "Done: $(date)"