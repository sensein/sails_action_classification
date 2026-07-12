#!/bin/bash
WORKSPACE="${WORKSPACE:-/home/aparnabg/orcd/pool/pyskl_workspace}"
PYSKL_ROOT=${WORKSPACE}/pyskl
LOG_DIR=${WORKSPACE}/train_logs
ENV_PATH=${WORKSPACE}/envs/pyskl
mkdir -p ${LOG_DIR}

submit_fusion() {
    local DATASET=$1
    local TIME=$2

    sbatch << SBATCH
#!/bin/bash
#SBATCH --job-name=fusion_${DATASET}
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_DIR}/fusion_${DATASET}_%j.out
#SBATCH --error=${LOG_DIR}/fusion_${DATASET}_%j.err

echo "=============================="
echo "Job    : fusion_${DATASET}"
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

echo "Extracting logits (val split) for ${DATASET}"
python fusion/extract_logits.py --dataset ${DATASET} --split val

echo "Extracting logits (test split) for ${DATASET}"
python fusion/extract_logits.py --dataset ${DATASET} --split test

echo "Training fusion MLP for ${DATASET}"
python fusion/train_mlp_fusion.py --dataset ${DATASET}

echo "Done: fusion_${DATASET} at \$(date)"
SBATCH

    echo "Submitted: fusion_${DATASET}"
}

# ================================================================
# Submit fusion jobs for both datasets
# ================================================================
submit_fusion "rmm"  "02:00:00"
submit_fusion "loco" "02:00:00"

echo ""
echo "All fusion jobs submitted."
echo "Monitor: squeue -u \$USER"
echo ""
echo "Outputs:"
echo "  work_dirs/fusion_rmm_val_logits.pkl"
echo "  work_dirs/fusion_rmm_test_logits.pkl"
echo "  work_dirs/fusion_mlp_rmm_best.pth"
echo "  work_dirs/fusion_loco_val_logits.pkl"
echo "  work_dirs/fusion_loco_test_logits.pkl"
echo "  work_dirs/fusion_mlp_loco_best.pth"
