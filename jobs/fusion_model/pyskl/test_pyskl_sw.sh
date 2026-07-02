#!/bin/bash
# test_all_new_sw.sh
WORKSPACE=/home/aparnabg/orcd/pool/pyskl_workspace
PYSKL_ROOT=${WORKSPACE}/pyskl
ENV_PATH=${WORKSPACE}/envs/pyskl

module purge
export PATH=/usr/bin:/bin:${ENV_PATH}/bin:$PATH
export CONDA_PREFIX=${ENV_PATH}
export LD_PRELOAD=${ENV_PATH}/lib/libstdc++.so.6
module load cuda/12.9.1

cd ${PYSKL_ROOT}

test_one() {
    local WORK_DIR=$1
    local CONFIG_PATTERN=$2

    CONFIG=$(ls ${WORK_DIR}/${CONFIG_PATTERN} 2>/dev/null | head -1)
    CKPT=$(ls ${WORK_DIR}/best_top1_acc_epoch_*.pth 2>/dev/null | head -1)

    if [[ -z "$CONFIG" || -z "$CKPT" ]]; then
        echo "[SKIP] ${WORK_DIR} — config or ckpt not found"
        return
    fi

    echo "Testing: ${WORK_DIR}"
    echo "  Config: ${CONFIG}"
    echo "  Ckpt:   ${CKPT}"

    bash tools/dist_test.sh ${CONFIG} ${CKPT} 1 \
        --out ${WORK_DIR}/test_pred.pkl \
        --eval top_k_accuracy mean_class_accuracy \
        2>&1 | tee ${WORK_DIR}/test_results.txt

    echo "Done: ${WORK_DIR}"
    echo "---"
}

# CTR-GCN locomotion/b
test_one "work_dirs/ctrgcn_locomotion_sw/b_s42"  "ctrgcn_b_locomotion_sw_s42.py"
test_one "work_dirs/ctrgcn_locomotion_sw/b_s123" "ctrgcn_b_locomotion_sw_s123.py"
test_one "work_dirs/ctrgcn_locomotion_sw/b_s456" "ctrgcn_b_locomotion_sw_s456.py"

# STGCN++ locomotion/b
test_one "work_dirs/stgcnpp_locomotion_sw/b_s42"  "stgcnpp_b_locomotion_sw_s42.py"
test_one "work_dirs/stgcnpp_locomotion_sw/b_s123" "stgcnpp_b_locomotion_sw_s123.py"
test_one "work_dirs/stgcnpp_locomotion_sw/b_s456" "stgcnpp_b_locomotion_sw_s456.py"

# PoseC3D RMM
test_one "work_dirs/posec3d_rmm_sw/joint_s42"  "posec3d_rmm_sw_s42.py"
test_one "work_dirs/posec3d_rmm_sw/joint_s123" "posec3d_rmm_sw_s123.py"
test_one "work_dirs/posec3d_rmm_sw/joint_s456" "posec3d_rmm_sw_s456.py"

# STGCN++ RMM/jm
test_one "work_dirs/stgcnpp_rmm_sw/jm_s42"  "stgcnpp_jm_rmm_sw_s42.py"
test_one "work_dirs/stgcnpp_rmm_sw/jm_s123" "stgcnpp_jm_rmm_sw_s123.py"
test_one "work_dirs/stgcnpp_rmm_sw/jm_s456" "stgcnpp_jm_rmm_sw_s456.py"

echo "All testing done!"