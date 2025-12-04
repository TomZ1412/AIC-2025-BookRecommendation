import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse

from recbole.quick_start import load_data_and_model
from recbole.data.interaction import Interaction

def argparse_():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='saved/BPR-Nov-21-2025_18-57-05.pth', help='模型文件')
    parser.add_argument('--submission_file', type=str, default='generated/submission_pre.csv', help='提交文件')
    return parser.parse_args()

MODEL_FILE = argparse_().model_file
SUBMISSION_FILE = argparse_().submission_file
TOP_K = 1
BATCH_SIZE = 256
try:
    config, model, dataset, _, _, _ = load_data_and_model(model_file=MODEL_FILE)
except Exception as e:
    print(f"❌ 无法加载模型文件 {MODEL_FILE}。请检查文件路径或解决 PyTorch 2.6+ 的 'weights_only' 兼容性问题。")
    print(f"错误详情: {e}")
    exit()

print(f"✅ 成功加载模型 {config['model']} 和数据集 {config['dataset']}。")
model.eval()
device = config['device']

num_users = dataset.user_num

all_user_ids_tensor = torch.arange(1, num_users, dtype=torch.long, device=device)

print(f"在交互数据中出现的用户总数: {len(all_user_ids_tensor)}")
print(f"内部 ID 范围: [1, {num_users-1}]")

# --- 4. 批量预测 Top-K 推荐 ---
recommended_list = []
total_users = len(all_user_ids_tensor)

print(f"开始为 {total_users} 位用户预测 Top-{TOP_K} 推荐 (批大小: {BATCH_SIZE})...")

for i in tqdm(range(0, total_users, BATCH_SIZE), desc="Generating Recommendations"):
    batch_user_ids_tensor = all_user_ids_tensor[i:i + BATCH_SIZE]

    interaction_batch = Interaction({config['USER_ID_FIELD']: batch_user_ids_tensor})

    interaction_batch.to(device)

    with torch.no_grad():
        prediction = model.full_sort_predict(interaction_batch)
    
    if prediction.dim() == 1:
        prediction = prediction.view(len(batch_user_ids_tensor), -1)
    
    topk_scores, topk_item_ids = torch.topk(prediction, k=TOP_K, dim=1)

    user_tokens_batch = dataset.id2token(dataset.uid_field, batch_user_ids_tensor.cpu().numpy())
    item_tokens_batch = dataset.id2token(dataset.iid_field, topk_item_ids.cpu().numpy().flatten())

    # 保存批次结果
    batch_df = pd.DataFrame({
        'user_id': user_tokens_batch,
        'item_id': item_tokens_batch
    })
    recommended_list.append(batch_df)

submission_df = pd.concat(recommended_list, ignore_index=True)

submission_df.columns = ['user_id', 'book_id']

submission_df.to_csv(
    SUBMISSION_FILE,
    index=False,
    encoding='utf-8'
)

print("-" * 30)
print(f"✅ 推荐结果已保存到: {os.path.abspath(SUBMISSION_FILE)}")
print(f"生成行数 (用户数): {len(submission_df)}")