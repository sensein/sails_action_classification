#!/bin/bash

SEEDS=(42 123 456)
TASKS=(loco rmm)
MODES=(clip fullvid twostage)

JOB_DIR=/jobs/action_model_testing/videomae2

for MODE in "${MODES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="vmae_${MODE}_${TASK}_${SEED}"
            echo "Submitting: $JOB_NAME"
            sbatch --job-name=${JOB_NAME} ${JOB_DIR}/videomae2_job.sh ${MODE} ${TASK} ${SEED}
        done
    done
done

echo ""
echo "All jobs submitted."