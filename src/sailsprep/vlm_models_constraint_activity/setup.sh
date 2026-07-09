#!/bin/bash
# Environment setup for VLM framework

set -e

CONDA_PATH="/home/aparnabg/orcd/scratch/miniconda3"
source "$CONDA_PATH/etc/profile.d/conda.sh"

echo "Creating environment: vlm_stable"
conda create -n vlm_stable python=3.10 -y

echo "Activating environment"
conda activate vlm_stable

echo "Installing PyTorch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing dependencies"
pip install transformers accelerate pandas scikit-learn nltk rouge-score av pillow qwen-vl-utils num2words protobuf sentencepiece

echo "Downloading NLTK data"
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "Verifying installation"
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__}'); print(f'Transformers: {transformers.__version__}')"

conda deactivate

echo "Setup complete"
echo "Activate with: conda activate vlm_stable"
