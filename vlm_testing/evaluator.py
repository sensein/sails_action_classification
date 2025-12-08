"""
Evaluation Metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from typing import List, Dict, Any


class VideoActivityEvaluator:
    """Evaluation metrics for both activity and context predictions"""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def calculate_bleu(self, predicted_activities, ground_truth_activities):
        """BLEU score for activity descriptions"""
        bleu_scores = []
        for pred, truth in zip(predicted_activities, ground_truth_activities):
            if pred and truth and not pd.isna(pred) and not pd.isna(truth):
                pred_tokens = str(pred).lower().split()
                truth_tokens = str(truth).lower().split()
                if len(pred_tokens) > 0 and len(truth_tokens) > 0:
                    score = sentence_bleu([truth_tokens], pred_tokens, 
                                         smoothing_function=self.smoothing)
                    bleu_scores.append(score)
        
        return {
            'mean_bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'std_bleu': np.std(bleu_scores) if bleu_scores else 0,
            'min_bleu': np.min(bleu_scores) if bleu_scores else 0,
            'max_bleu': np.max(bleu_scores) if bleu_scores else 0,
            'count': len(bleu_scores)
        }
    
    def calculate_rouge(self, predicted_activities, ground_truth_activities):
        """ROUGE score for activity descriptions"""
        rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
        
        for pred, truth in zip(predicted_activities, ground_truth_activities):
            if pred and truth and not pd.isna(pred) and not pd.isna(truth):
                try:
                    pred_str = str(pred).lower().strip()
                    truth_str = str(truth).lower().strip()
                    if len(pred_str) > 0 and len(truth_str) > 0:
                        scores = self.scorer.score(truth_str, pred_str)
                        rouge_scores['rouge-1'].append(scores['rouge1'].fmeasure)
                        rouge_scores['rouge-2'].append(scores['rouge2'].fmeasure)
                        rouge_scores['rouge-l'].append(scores['rougeL'].fmeasure)
                except Exception as e:
                    continue
        
        return {
            'rouge-1': np.mean(rouge_scores['rouge-1']) if rouge_scores['rouge-1'] else 0,
            'rouge-2': np.mean(rouge_scores['rouge-2']) if rouge_scores['rouge-2'] else 0,
            'rouge-l': np.mean(rouge_scores['rouge-l']) if rouge_scores['rouge-l'] else 0,
            'count': len(rouge_scores['rouge-1'])
        }
    
    def calculate_exact_match(self, predicted_activities, ground_truth_activities):
        """Exact match rate for activities"""
        valid_pairs = [(str(p).lower().strip(), str(g).lower().strip()) 
                      for p, g in zip(predicted_activities, ground_truth_activities)
                      if p and g and not pd.isna(p) and not pd.isna(g)]
        
        if not valid_pairs:
            return 0.0
        
        matches = sum(1 for pred, truth in valid_pairs if pred == truth)
        return matches / len(valid_pairs)
    
    def calculate_word_overlap(self, predicted_activities, ground_truth_activities):
        """Average word overlap between predicted and ground truth activities"""
        overlaps = []
        for pred, truth in zip(predicted_activities, ground_truth_activities):
            if pred and truth and not pd.isna(pred) and not pd.isna(truth):
                pred_words = set(str(pred).lower().split())
                truth_words = set(str(truth).lower().split())
                
                if len(truth_words) > 0:
                    overlap = len(pred_words & truth_words) / len(truth_words)
                    overlaps.append(overlap)
        
        return {
            'mean_overlap': np.mean(overlaps) if overlaps else 0,
            'std_overlap': np.std(overlaps) if overlaps else 0,
            'count': len(overlaps)
        }
    
    def calculate_context_accuracy(self, predicted_contexts, ground_truth_contexts):
        """Overall accuracy for context classification"""
        valid_pairs = []
        for p, g in zip(predicted_contexts, ground_truth_contexts):
            if p is not None and g is not None and not pd.isna(p) and not pd.isna(g):
                try:
                    p_int = int(p)
                    g_int = int(g)
                    valid_pairs.append((p_int, g_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return 0.0
        
        pred, truth = zip(*valid_pairs)
        return accuracy_score(truth, pred)
    
    def calculate_context_f1(self, predicted_contexts, ground_truth_contexts):
        """F1 scores for context classification"""
        valid_pairs = []
        for p, g in zip(predicted_contexts, ground_truth_contexts):
            if p is not None and g is not None and not pd.isna(p) and not pd.isna(g):
                try:
                    p_int = int(p)
                    g_int = int(g)
                    valid_pairs.append((p_int, g_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return {'macro_f1': 0.0, 'weighted_f1': 0.0}
        
        pred, truth = zip(*valid_pairs)
        
        return {
            'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
            'weighted_f1': f1_score(truth, pred, average='weighted', zero_division=0)
        }
    
    def get_classification_report(self, predicted_contexts, ground_truth_contexts):
        """Detailed classification report"""
        valid_pairs = []
        for p, g in zip(predicted_contexts, ground_truth_contexts):
            if p is not None and g is not None and not pd.isna(p) and not pd.isna(g):
                try:
                    p_int = int(p)
                    g_int = int(g)
                    valid_pairs.append((p_int, g_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return "No valid predictions to evaluate"
        
        pred, truth = zip(*valid_pairs)

        target_names = [
            '1: special occasion',
            '2: general social communication interaction',
            '3: motor play',
            '4: daily routine',
            '5: toy play',
            '6: social routine',
            '7: other',
            '8: book share'
        ]
        
        return classification_report(truth, pred, zero_division=0, 
                                    labels=[1, 2, 3, 4, 5, 6, 7, 8],
                                    target_names=target_names)
    
    def get_confusion_matrix(self, predicted_contexts, ground_truth_contexts):
        """Confusion matrix for context classification"""
        valid_pairs = []
        for p, g in zip(predicted_contexts, ground_truth_contexts):
            if p is not None and g is not None and not pd.isna(p) and not pd.isna(g):
                try:
                    p_int = int(p)
                    g_int = int(g)
                    valid_pairs.append((p_int, g_int))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return None
        
        pred, truth = zip(*valid_pairs)
        return confusion_matrix(truth, pred, labels=[1, 2, 3, 4, 5, 6, 7, 8])
    
    def evaluate_all(self, predicted_activities, ground_truth_activities, 
                     predicted_contexts, ground_truth_contexts):

        results = {
            'activity_metrics': {
                'bleu': self.calculate_bleu(predicted_activities, ground_truth_activities),
                'rouge': self.calculate_rouge(predicted_activities, ground_truth_activities),
                'exact_match': self.calculate_exact_match(predicted_activities, ground_truth_activities),
                'word_overlap': self.calculate_word_overlap(predicted_activities, ground_truth_activities)
            },
            'context_metrics': {
                'accuracy': self.calculate_context_accuracy(predicted_contexts, ground_truth_contexts),
                'f1_scores': self.calculate_context_f1(predicted_contexts, ground_truth_contexts),
                'classification_report': self.get_classification_report(predicted_contexts, ground_truth_contexts),
                'confusion_matrix': self.get_confusion_matrix(predicted_contexts, ground_truth_contexts)
            }
        }
        
        return results
