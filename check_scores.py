import json

# 1. 读取标准集 (gold) 和 预测集 (pred)
gold_data = {}
with open('eval_draft.jsonl', 'r', encoding='utf-8') as f: #jsonl和json格式不同，不能使用json.load()，需要逐行读取并解析
    for line in f:
        if line.strip():
            item = json.loads(line) #逐行读取并解析成字典对象
            gold_data[item['id']] = item

pred_data = {}
with open('pred.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            item = json.loads(line) #逐行读取并解析成字典对象
            pred_data[item['id']] = item

# 2. 分类收集分数
pos_scores = []  # 存放 answerable=True 的分数
neg_scores = []  # 存放 answerable=False 的分数

for qid, gold_item in gold_data.items():
    if qid in pred_data:
        score = pred_data[qid]['best_score']
        # 注意判断布尔值
        if gold_item['answerable'] is True: #将可答和拒答的分数分别收集到不同的列表中，方便后续统计分析
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


#统计数据分布的目的是为分析模型拒答率的合理性，观察模型在应该回答和应该拒答的样本上的分数分布情况，评估模型的区分能力和拒答策略的有效性。通过比较两类样本的分数分布，可以判断模型是否能够正确识别出哪些问题应该回答，哪些问题应该拒答，从而优化模型的性能和用户体验。