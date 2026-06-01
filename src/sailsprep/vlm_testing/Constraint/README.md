# V-JEPA2 Constraint Prediction with Temporal Windowing

Complete pipeline for training V-JEPA2 to predict child constraint status from videos using temporal windowing and majority voting.

## Overview

This package predicts:
1. **Constraint Status**: y (constrained) / n (not constrained) / partial
2. **Constraint Type**: highchair, carseat, stroller, etc. (only if constrained)

**Method**: Temporal windowing with majority voting
- Divides video into N windows
- Samples frames from each window
- Gets prediction for each window  
- Final prediction = majority vote across windows

### Prerequisites
- Python 3.10+
- Your CSV file with columns: `BidsProcessed`, `Child_constrained`, `Constraint_type`



### **Step 1: Prepare Data**

```bash
python prepare_data.py \
    --csv your_data.csv \
    --video_col BidsProcessed \
    --constrained_col Child_constrained \
    --type_col Constraint_type \
    --val_split 0.3 \
    --test_split 0.3 \
    --output prepared_data.pkl
```

**Output:**
- `prepared_data.pkl` with train/val/test splits 

---

### **Step 2: Train Model**

#### **Option A: Without Windowing (Faster)**
```bash
python finetune_vjepa_windowing.py \
    --data prepared_data.pkl \
    --model_cache ./models \
    --output ./finetuned_vjepa \
    --epochs 5 \
    --batch_size 32 \
    --lr 1e-4 \
    --num_workers 8
```

#### **Option B: With Windowing (More Accurate)**
```bash
python finetune_vjepa_windowing.py \
    --data prepared_data.pkl \
    --model_cache ./models \
    --output ./finetuned_vjepa \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-4 \
    --num_workers 8 \
    --use_windowing \
    --num_windows 3
```

**Output:**
- `./finetuned_vjepa/best_model.pt` - Trained model
- `./finetuned_vjepa/training_history.json` - Training metrics

---

### **Step 3: Evaluate on Test Set**

```bash
python evaluate_vjepa_windowing.py \
    --data prepared_data.pkl \
    --model ./finetuned_vjepa \
    --split test \
    --output ./test_results
```

**Output:**
- `./test_results/test_predictions.csv` - All predictions
- `./test_results/test_metrics.json` - Accuracy, F1, confusion matrix

---

