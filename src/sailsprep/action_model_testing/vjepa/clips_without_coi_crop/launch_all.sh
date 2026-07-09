#!/bin/bash
# ============================================================
# Master launcher
# Usage: bash launch_all.sh
#
# What it does:
#   1. Submits extract job
#   2. Submits seed array job with --dependency=afterok:<extract_job_id>
#      so seeds only start after features are successfully extracted
# ============================================================

LOG_DIR="/orcd/data/satra/002/projects/SAILS/vjepa_features/models_output_seeds/clips/vjepa/logs"
mkdir -p $LOG_DIR

CODE_DIR="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjeap_full_video"

# --- Step 1: Submit feature extraction ---
EXTRACT_JOB=$(sbatch --parsable ${CODE_DIR}/submit_extract.sh)
echo "Submitted feature extraction job: $EXTRACT_JOB"

# --- Step 2: Submit seed training jobs (depend on extraction finishing) ---
SEED_JOB=$(sbatch --parsable \
    --dependency=afterok:${EXTRACT_JOB} \
    ${CODE_DIR}/submit_seeds.sh)
echo "Submitted seed training jobs (array): $SEED_JOB"
echo "  -> Seeds 42, 456, 123 will start after job $EXTRACT_JOB completes"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/extract_${EXTRACT_JOB}.out"
echo "  tail -f ${LOG_DIR}/probe_${SEED_JOB}_0.out   # seed 42"
echo "  tail -f ${LOG_DIR}/probe_${SEED_JOB}_1.out   # seed 456"
echo "  tail -f ${LOG_DIR}/probe_${SEED_JOB}_2.out   # seed 123"
