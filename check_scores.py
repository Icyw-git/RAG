import json

# 1. 读取标准集 (gold) 和 预测集 (pred)
gold_data = {}
with open('eval_draft.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            gold_data[item['id']] = item

pred_data = {}
with open('pred.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            pred_data[item['id']] = item

# 2. 分类收集分数
pos_scores = []  # 存放 answerable=True 的分数
neg_scores = []  # 存放 answerable=False 的分数

for qid, gold_item in gold_data.items():
    if qid in pred_data:
        score = pred_data[qid]['best_score']
        # 注意判断布尔值
        if gold_item['answerable'] is True:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

# 3. 打印统计结果
def print_stats(name, scores):
    if not scores:
        print(f"{name}: 0 条数据")
        return
    min_s = min(scores)
    max_s = max(scores)
    avg_s = sum(scores) / len(scores)
    print(f"{name} (共 {len(scores)} 条): Min = {min_s:.4f} | Avg = {avg_s:.4f} | Max = {max_s:.4f}")

print("=== 检索分数分布统计 ===")
print_stats("answerable=True (应该回答的)", pos_scores)
print_stats("answerable=False (应该拒答的)", neg_scores)