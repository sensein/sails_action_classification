#!/bin/bash
#SBATCH --job-name=test_sw
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --array=0-2

WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
ENV_PATH=${WORKSPACE}/envs/pyskl

MODEL=$1
TASK=$2

SEEDS=(42 123 456)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}


module purge
module load miniforge
conda deactivate

source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
export PATH=/usr/bin:/bin:${ENV_PATH}/bin:$PATH
export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6


cd ${PYSKL_ROOT}

if [[ "$MODEL" == "posec3d" ]]; then
    MODALITY="joint"
elif [[ "$MODEL" == "ctrgcn" ]]; then
    MODALITY="jm"
fi

WORK_DIR="work_dirs/${MODEL}_${TASK}_sw/${MODALITY}_s${SEED}"
CONFIG_FILE=$(ls ${WORK_DIR}/*.py 2>/dev/null | head -1)
CKPT=$(ls ${WORK_DIR}/best_top1_acc_epoch_*.pth 2>/dev/null | head -1)

if [[ -z "$CKPT" ]]; then
    echo "No checkpoint in ${WORK_DIR}"
    exit 1
fi

echo "Testing: ${MODEL} ${TASK} seed=${SEED}"
echo "Config: ${CONFIG_FILE}"
echo "Ckpt: ${CKPT}"

bash tools/dist_test.sh ${CONFIG_FILE} ${CKPT} 1 \
    --out ${WORK_DIR}/test_pred.pkl \
    --eval top_k_accuracy mean_class_accuracy \
    2>&1 | tee ${WORK_DIR}/test_results.txt

echo "Done: $(date)"