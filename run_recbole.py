import pandas as pd
import numpy as np
import os
import time
import argparse

# RecBole 核心库
from recbole.quick_start import run_recbole

# --- F1 计算函数 ---
def calculate_f1_score(best_result):
    """
    从 RecBole 的 best_result 字典中提取 Precision 和 Recall 
    并计算 F1 Score。
    """
    f1_results = {}
    # import pdb;pdb.set_trace()
    # 遍历所有结果，寻找 Precision@K
    for key, prec_val in best_result.items():
        if key.startswith('precision@'):
            # 提取 K 值
            k_value = key.split('@')[1]
            recall_key = f'recall@{k_value}'
            recall_val = best_result.get(recall_key)
            
            if recall_val is not None:
                # 核心 F1 计算公式：2 * (P * R) / (P + R)
                if (prec_val + recall_val) == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (prec_val * recall_val) / (prec_val + recall_val)
                
                f1_results[f'F1@{k_value}'] = f1_score
                
    return f1_results
def argparse_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='pre', help='比赛阶段')
    return parser.parse_args()

phase = argparse_().phase
config_file_list = [f'config_bpr_{phase}_part.yaml']
parameter_dict = {
    'config_file_list': config_file_list
}
print("Starting RecBole training with BPR model...")

result = run_recbole(**parameter_dict)
print("\n--- Training Completed ---")
print("Evaluation Results:")
print(result['best_valid_score'])

print("\n--- F1 Score Results ---")

f1_scores = calculate_f1_score(result['test_result'])

if f1_scores:
    for k, f1 in f1_scores.items():
        print(f"{k}: {f1:.6f}")
else:
    print("WARNING: Could not calculate F1 Score. Please ensure 'Precision' and 'Recall' are included in metrics in config.yaml.")