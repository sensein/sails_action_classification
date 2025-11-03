#!/bin/bash
#SBATCH --job-name=merge_cleanup
#SBATCH --output=logs/merge_cleanup_%j.out
#SBATCH --error=logs/merge_cleanup_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=2G

# Clean up old logs before running
echo "Cleaning up old logs..."
rm -rf logs
mkdir -p logs

OUTPUT_DIR=$(poetry run python -c "import yaml; with open('configs/config_bids_convertor.yaml') as f: print(yaml.safe_load(f)['output_dir'])")
MERGED_DIR="$OUTPUT_DIR"

mkdir -p "$MERGED_DIR"

echo "Merging logs from numbered folders under $OUTPUT_DIR"
echo "Started at $(date)"

merged_processed="$MERGED_DIR/all_processed.json"
merged_failed="$MERGED_DIR/all_failed.json"

# Create empty lists if not exist
echo "[]" > "$merged_processed"
echo "[]" > "$merged_failed"

# Load jq (if not already available)
module load jq 2>/dev/null || true

for folder in "$OUTPUT_DIR"/*/; do
    foldername=$(basename "$folder")

    if [[ "$foldername" =~ ^[0-9]+$ ]]; then
        echo "Merging from folder: $foldername"
        if [[ -f "$folder/processing_log.json" ]]; then
            jq -s 'add' "$merged_processed" "$folder/processing_log.json" > tmp.json && mv tmp.json "$merged_processed"
        fi
        if [[ -f "$folder/not_processed.json" ]]; then
            jq -s 'add' "$merged_failed" "$folder/not_processed.json" > tmp.json && mv tmp.json "$merged_failed"
        fi
    fi
done

echo "Merged logs saved in: $MERGED_DIR"
echo "Now cleaning up numbered folders..."

# Delete only folders with numeric names (avoid final_bids-dataset)
for folder in "$OUTPUT_DIR"/*/; do
    foldername=$(basename "$folder")
    if [[ "$foldername" =~ ^[0-9]+$ ]]; then
        echo "Deleting temporary folder: $foldername"
        rm -rf "$folder"
    else
        echo "Skipping non-numeric folder: $foldername"
    fi
done

echo "Cleanup complete at $(date)"

# --- Run final Python merge ---
echo "Running final Python merge and participant file creation..."
poetry run python -c "from src.BIDS_convertor import merge_subjects, create_participants_file; merge_subjects(); create_participants_file()"
echo "Final BIDS merge and participant file creation complete ✅"
