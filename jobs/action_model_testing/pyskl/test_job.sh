#!/bin/bash
WORKSPACE="${WORKSPACE:-/home/aparnabg/orcd/pool/pyskl_workspace}"
PYSKL_ROOT=${WORKSPACE}/pyskl
LOG_DIR=${WORKSPACE}/train_logs
ENV_PATH=${WORKSPACE}/envs/pyskl
mkdir -p ${LOG_DIR}

submit_test() {
    local JOB_NAME=$1
    local CONFIG=$2
    local WORK_DIR=$3
    local SEED=$4

    # Find best checkpoint
    CKPT=$(ls ${PYSKL_ROOT}/${WORK_DIR}/best_top1_acc_epoch_*.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "SKIPPING ${JOB_NAME}_s${SEED} — no checkpoint found in ${WORK_DIR}"
        return
    fi

    echo "Submitting test: ${JOB_NAME}_s${SEED}, ckpt: $(basename $CKPT)"

    sbatch << SBATCH
#!/bin/bash
#SBATCH --job-name=test_${JOB_NAME}_s${SEED}
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=${LOG_DIR}/test_${JOB_NAME}_s${SEED}_%j.out
#SBATCH --error=${LOG_DIR}/test_${JOB_NAME}_s${SEED}_%j.err

module purge
export PATH=/usr/bin:/bin:${ENV_PATH}/bin:\$PATH
export CONDA_PREFIX=${ENV_PATH}
export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6
module load cuda/12.9.1

cd ${PYSKL_ROOT}

# Patch config to use test split
TEST_CONFIG=/tmp/test_${JOB_NAME}_s${SEED}.py
cp ${CONFIG} \${TEST_CONFIG}
python - << PYEOF
with open('\${TEST_CONFIG}', 'r') as f:
    content = f.read()
# Change work_dir to seed-specific dir
import re
content = re.sub(r"work_dir = '(.*?)'", r"work_dir = '${WORK_DIR}'", content)
with open('\${TEST_CONFIG}', 'w') as f:
    f.write(content)
PYEOF

echo "Testing: ${JOB_NAME}_s${SEED}"
echo "Config : \${TEST_CONFIG}"
echo "Ckpt   : ${CKPT}"

bash tools/dist_test.sh \${TEST_CONFIG} ${CKPT} 1 \
    --out ${PYSKL_ROOT}/${WORK_DIR}/test_pred.pkl \
    --eval top_k_accuracy mean_class_accuracy \
    2>&1 | tee ${PYSKL_ROOT}/${WORK_DIR}/test_results.txt

echo "Done testing ${JOB_NAME}_s${SEED}"
SBATCH
}

# ================================================================
# All models x 3 seeds
# ================================================================
for SEED in 123 42 456; do
    # RMM
    submit_test "stgcnpp_rmm_j"  "configs/custom/stgcnpp_rmm/j.py"     "work_dirs/stgcnpp_rmm/j_s${SEED}"   ${SEED}
    submit_test "stgcnpp_rmm_b"  "configs/custom/stgcnpp_rmm/b.py"     "work_dirs/stgcnpp_rmm/b_s${SEED}"   ${SEED}
    submit_test "stgcnpp_rmm_jm" "configs/custom/stgcnpp_rmm/jm.py"    "work_dirs/stgcnpp_rmm/jm_s${SEED}"  ${SEED}
    submit_test "stgcnpp_rmm_bm" "configs/custom/stgcnpp_rmm/bm.py"    "work_dirs/stgcnpp_rmm/bm_s${SEED}"  ${SEED}
    submit_test "ctrgcn_rmm_j"   "configs/custom/ctrgcn_rmm/j.py"      "work_dirs/ctrgcn_rmm/j_s${SEED}"    ${SEED}
    submit_test "ctrgcn_rmm_b"   "configs/custom/ctrgcn_rmm/b.py"      "work_dirs/ctrgcn_rmm/b_s${SEED}"    ${SEED}
    submit_test "ctrgcn_rmm_jm"  "configs/custom/ctrgcn_rmm/jm.py"     "work_dirs/ctrgcn_rmm/jm_s${SEED}"   ${SEED}
    submit_test "ctrgcn_rmm_bm"  "configs/custom/ctrgcn_rmm/bm.py"     "work_dirs/ctrgcn_rmm/bm_s${SEED}"   ${SEED}
    submit_test "posec3d_rmm"    "configs/custom/posec3d_rmm/joint.py"  "work_dirs/posec3d_rmm/joint_s${SEED}" ${SEED}

    # LOCO
    submit_test "stgcnpp_loco_j"  "configs/custom/stgcnpp_loco/j.py"    "work_dirs/stgcnpp_loco/j_s${SEED}"   ${SEED}
    submit_test "stgcnpp_loco_b"  "configs/custom/stgcnpp_loco/b.py"    "work_dirs/stgcnpp_loco/b_s${SEED}"   ${SEED}
    submit_test "stgcnpp_loco_jm" "configs/custom/stgcnpp_loco/jm.py"   "work_dirs/stgcnpp_loco/jm_s${SEED}"  ${SEED}
    submit_test "stgcnpp_loco_bm" "configs/custom/stgcnpp_loco/bm.py"   "work_dirs/stgcnpp_loco/bm_s${SEED}"  ${SEED}
    submit_test "ctrgcn_loco_j"   "configs/custom/ctrgcn_loco/j.py"     "work_dirs/ctrgcn_loco/j_s${SEED}"    ${SEED}
    submit_test "ctrgcn_loco_b"   "configs/custom/ctrgcn_loco/b.py"     "work_dirs/ctrgcn_loco/b_s${SEED}"    ${SEED}
    submit_test "ctrgcn_loco_jm"  "configs/custom/ctrgcn_loco/jm.py"    "work_dirs/ctrgcn_loco/jm_s${SEED}"   ${SEED}
    submit_test "ctrgcn_loco_bm"  "configs/custom/ctrgcn_loco/bm.py"    "work_dirs/ctrgcn_loco/bm_s${SEED}"   ${SEED}
    submit_test "posec3d_loco"    "configs/custom/posec3d_loco/joint.py" "work_dirs/posec3d_loco/joint_s${SEED}" ${SEED}
done

echo ""
echo "All test jobs submitted. Monitor: squeue -u \$USER"
