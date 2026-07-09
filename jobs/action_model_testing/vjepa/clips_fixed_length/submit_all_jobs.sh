#!/bin/bash
# Usage: bash submit_all.sh

SCRIPT_DIR="${SCRIPT_DIR:-/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/vjepa_crop}"

echo "Submitting all 6 jobs (all reuse existing feature cache)..."
for LABEL in loco rmm; do
  for SEED in 42 123 456; do
    JID=$(sbatch --parsable "$SCRIPT_DIR/vjepa_clip_level_ablation.sh" $LABEL $SEED)
    echo "  $LABEL seed $SEED -> job $JID"
  done
done
echo ""
echo "All 6 jobs submitted. Check with: squeue -u $USER"