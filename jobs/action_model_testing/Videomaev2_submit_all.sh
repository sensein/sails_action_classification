#!/bin/bash

SEEDS=(42 123 456)
TASKS=(loco rmm)
MODES=(clip fullvid twostage)

SCRIPT_DIR=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/Videomaev2/clip

for MODE in "${MODES[@]}"; do
    for TASK in "${TASKS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            JOB_NAME="vmae_${MODE}_${TASK}_${SEED}"
            echo "Submitting: $JOB_NAME"
            sbatch --job-name=${JOB_NAME} ${SCRIPT_DIR}/job.sh ${MODE} ${TASK} ${SEED}
        done
    done
done

echo ""
echo "All jobs submitted."