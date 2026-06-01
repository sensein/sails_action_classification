#!/bin/bash
# submit_iv2_all.sh
# Submits all seed jobs and then a dependent aggregation job.
#
# Usage:
#   bash submit_iv2_all.sh

SCRIPT_DIR=/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/InternVideo_v2/clip

# Submit the array job (0-5 = loco x 3 seeds + rmm x 3 seeds).
ARRAY_JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/job_clip.sh")
echo "Submitted array job: ${ARRAY_JOB_ID}"

# Submit aggregation after ALL array tasks complete.
sbatch \
    --dependency=afterok:"${ARRAY_JOB_ID}" \
    --partition=ou_bcs_normal \
    --job-name=IV2_aggregate \
    --time=00:10:00 \
    --mem=4G \
    --cpus-per-task=1 \
    --output="${SCRIPT_DIR}/logs/IV2_agg_%j.out" \
    --error="${SCRIPT_DIR}/logs/IV2_agg_%j.err" \
    --wrap="
        module load miniforge
        source /home/aparnabg/orcd/pool/miniconda3/etc/profile.d/conda.sh
        conda activate iv2
        cd ${SCRIPT_DIR}
        python aggregate_iv2_seeds.py --task loco
        python aggregate_iv2_seeds.py --task rmm
    "

echo "Aggregation job submitted (runs after ${ARRAY_JOB_ID} completes)"