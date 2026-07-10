#!/bin/bash
# job.sh — submit all VideoMAE weighted models
# Run: bash job.sh
 
BASE="/home/aparnabg/orcd/scratch/all_project_files/action_sota_models/Videomaev2/class_weights"
LOG_DIR="$BASE/logs"
mkdir -p $LOG_DIR   # create logs dir BEFORE sbatch tries to write there

for MODEL in vit_s vit_b vit_l vit_h vit_g; do

    echo "Submitting: $MODEL"

    sbatch <<ENDJOB
#!/bin/bash
#SBATCH --job-name=weighted_${MODEL}
#SBATCH --partition=pi_satra
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=${LOG_DIR}/weighted_${MODEL}_%j.out
#SBATCH --error=${LOG_DIR}/weighted_${MODEL}_%j.err

source /home/aparnabg/orcd/scratch/miniconda3/etc/profile.d/conda.sh
conda activate Videomae_env
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export MODEL=${MODEL}
export MASTER_CSV="/home/aparnabg/orcd/scratch/all_project_files/splits_loco_cut-clips_v2.csv"
export OUTPUT_DIR="${BASE}/output/locomotion_${MODEL}_weighted"

echo "=========================================="
echo "Job ID : \$SLURM_JOB_ID"
echo "Node   : \$SLURMD_NODENAME"
echo "GPU    : \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "MODEL  : ${MODEL}"
echo "Start  : \$(date)"
echo "=========================================="

cd ${BASE}
python finetune_clips_weighted.py

echo "=========================================="
echo "End: \$(date)"
echo "=========================================="
ENDJOB

    echo "  → Submitted $MODEL"
done

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Logs in: $LOG_DIR"