#!/bin/bash
# Usage: bash submit_all.sh

JOB_DIR="${JOB_DIR:-/jobs/action_model_testing/vjepa/clips_fixed_length}"

echo "Submitting all 6 jobs (all reuse existing feature cache)..."
for LABEL in loco rmm; do
  for SEED in 42 123 456; do
    JID=$(sbatch --parsable "$JOB_DIR/vjepa_clip_level_ablation.sh" $LABEL $SEED)
    echo "  $LABEL seed $SEED -> job $JID"
  done
done
echo ""
echo "All 6 jobs submitted. Check with: squeue -u $USER"