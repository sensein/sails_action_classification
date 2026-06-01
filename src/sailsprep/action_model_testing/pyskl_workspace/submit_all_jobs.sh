#!/bin/bash
WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
LOG_DIR=${WORKSPACE}/train_logs
ENV_PATH=${WORKSPACE}/envs/pyskl
mkdir -p ${LOG_DIR}

submit_job() {
    local JOB_NAME=$1
    local CONFIG=$2
    local TIME=$3
    local SEED=$4

    sbatch << SBATCH
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}_s${SEED}
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/${JOB_NAME}_s${SEED}_%j.out
#SBATCH --error=${LOG_DIR}/${JOB_NAME}_s${SEED}_%j.err

echo "=============================="
echo "Job    : ${JOB_NAME}_s${SEED}"
echo "Config : ${CONFIG}"
echo "Seed   : ${SEED}"
echo "Host   : \$(hostname)"
echo "Start  : \$(date)"
echo "=============================="

module purge
export PATH=/usr/bin:/bin:${ENV_PATH}/bin:\$PATH
export CONDA_PREFIX=${ENV_PATH}
export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6
module load cuda/12.9.1

echo "Python  : \$(which python)"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

nvidia-smi | head -10

cd ${PYSKL_ROOT}

# Each seed gets its own work_dir so results don't overwrite each other
# We do this by temporarily patching work_dir in a copied config
SEED_CONFIG=/tmp/${JOB_NAME}_s${SEED}.py
cp ${CONFIG} \${SEED_CONFIG}
sed -i "s|work_dir = '|work_dir = '|" \${SEED_CONFIG}
# Replace work_dir to include seed
python - << PYEOF
import re
with open('\${SEED_CONFIG}', 'r') as f:
    content = f.read()
content = re.sub(r"work_dir = '(.*?)'", r"work_dir = '\1_s${SEED}'", content)
with open('\${SEED_CONFIG}', 'w') as f:
    f.write(content)
PYEOF

echo "Work dir patched for seed ${SEED}"
cat \${SEED_CONFIG} | grep work_dir

bash tools/dist_train.sh \${SEED_CONFIG} 1 --validate --test-last --test-best --seed ${SEED}

echo "Done: ${JOB_NAME}_s${SEED} at \$(date)"
SBATCH

    echo "Submitted: ${JOB_NAME}_s${SEED}"
}

# ================================================================
# Submit all models x 3 seeds = 54 jobs total
# ================================================================
for SEED in 123 42 456; do

    echo "--- Submitting seed ${SEED} ---"

    # RMM (4 classes)
    submit_job "stgcnpp_rmm_j"  "configs/custom/stgcnpp_rmm/j.py"     "04:00:00" ${SEED}
    submit_job "stgcnpp_rmm_b"  "configs/custom/stgcnpp_rmm/b.py"     "04:00:00" ${SEED}
    submit_job "stgcnpp_rmm_jm" "configs/custom/stgcnpp_rmm/jm.py"    "04:00:00" ${SEED}
    submit_job "stgcnpp_rmm_bm" "configs/custom/stgcnpp_rmm/bm.py"    "04:00:00" ${SEED}
    submit_job "ctrgcn_rmm_j"   "configs/custom/ctrgcn_rmm/j.py"      "04:00:00" ${SEED}
    submit_job "ctrgcn_rmm_b"   "configs/custom/ctrgcn_rmm/b.py"      "04:00:00" ${SEED}
    submit_job "ctrgcn_rmm_jm"  "configs/custom/ctrgcn_rmm/jm.py"     "04:00:00" ${SEED}
    submit_job "ctrgcn_rmm_bm"  "configs/custom/ctrgcn_rmm/bm.py"     "04:00:00" ${SEED}
    submit_job "posec3d_rmm"    "configs/custom/posec3d_rmm/joint.py"  "06:00:00" ${SEED}

    # LOCO (5 classes)
    submit_job "stgcnpp_loco_j"  "configs/custom/stgcnpp_loco/j.py"    "06:00:00" ${SEED}
    submit_job "stgcnpp_loco_b"  "configs/custom/stgcnpp_loco/b.py"    "06:00:00" ${SEED}
    submit_job "stgcnpp_loco_jm" "configs/custom/stgcnpp_loco/jm.py"   "06:00:00" ${SEED}
    submit_job "stgcnpp_loco_bm" "configs/custom/stgcnpp_loco/bm.py"   "06:00:00" ${SEED}
    submit_job "ctrgcn_loco_j"   "configs/custom/ctrgcn_loco/j.py"     "06:00:00" ${SEED}
    submit_job "ctrgcn_loco_b"   "configs/custom/ctrgcn_loco/b.py"     "06:00:00" ${SEED}
    submit_job "ctrgcn_loco_jm"  "configs/custom/ctrgcn_loco/jm.py"    "06:00:00" ${SEED}
    submit_job "ctrgcn_loco_bm"  "configs/custom/ctrgcn_loco/bm.py"    "06:00:00" ${SEED}
    submit_job "posec3d_loco"    "configs/custom/posec3d_loco/joint.py" "06:00:00" ${SEED}

done

echo ""
echo "All 54 jobs submitted (18 models x 3 seeds)"
echo "Monitor: squeue -u \$USER"
echo ""
echo "Work dirs will be structured as:"
echo "  work_dirs/stgcnpp_rmm/j_s123/"
echo "  work_dirs/stgcnpp_rmm/j_s42/"
echo "  work_dirs/stgcnpp_rmm/j_s456/"
echo "  ... etc"
