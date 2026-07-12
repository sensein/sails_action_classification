"""Shared class-label configuration for the master-CSV-based fine-tuning scripts
(ablation study, class_weight, without_classweight).
"""

# CSV folder names -> internal class names
CSV_CLASS_TO_INTERNAL = {
    "Walking": "walk",
    "Cruising": "cruise",
    "Crawling": "crawl",
    "Vehicle": "vehicle",
    "Running": "run",
}

ACTION_CLASSES = ["walk", "cruise", "crawl", "vehicle", "run"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(ACTION_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}
