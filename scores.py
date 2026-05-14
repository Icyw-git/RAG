import json
from eval_logger import EvalLogger

gold_path = "eval_draft1.jsonl"
pred_path = "pred_hybrid.jsonl"
k = 5

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line)) # 加载JSONL文件，每个样本占一行
    return items

gold = {x["id"]: x for x in load_jsonl(gold_path)} # 加载标准答案，每个样本占一行
pred = {x["id"]: x for x in load_jsonl(pred_path)} # 加载预测结果，每个样本占一行

# ====== 1) Retrieval: source/chunk 命中率（只对 answerable=True 的样本） ======
total_ans = 0
hit_source_at1 = 0 # 检索到的文档来源命中率
hit_source_atk = 0 # 检索到的文档来源命中率，k=5，即检索到的文档来源中，是否有任何文档来源与标准答案中的文档来源匹配
hit_chunk_at1 = 0 # 检索到的文档chunk命中率
hit_chunk_atk = 0 # 检索到的文档chunk命中率，k=5，即检索到的文档chunk中，是否有任何文档chunk与标准答案中的文档chunk匹配

# ====== 2) Refusal: 拒答分类指标（对所有样本） ======
TP = FP = FN = TN = 0  # 这里的“正类”定义为 should_refuse=True

badcases = [] # 存储错误的样本，包括错误的拒绝回答和错误的接受回答

for qid, g in gold.items():
    p = pred.get(qid)
    if p is None: # 如果预测结果中缺少该样本的预测结果，记录为一个错误样本，类型为“missing_pred”，并继续处理下一个样本
        badcases.append({"id": qid, "type": "missing_pred"}) # 缺少预测结果
        continue

    should_refuse = (g["answerable"] is False) # 如果标准答案中，该样本不可回答，且预测结果中，该样本被拒绝回答
    pred_refuse = (p["refused"] is True)  or ('抱歉' in p.get('answer','')) # 如果预测结果中，该样本被拒绝回答，或者模型生成的回答中包含“抱歉”等拒绝回答的提示语

    if should_refuse and pred_refuse: # 正确拒绝回答
        TP += 1
    elif (not should_refuse) and pred_refuse: # 错误拒绝回答
        FP += 1 # 错误拒绝回答的样本数
        badcases.append({"id": qid, "type": "over_refuse", "question": g["question"], "best_score": p["best_score"]})
    elif should_refuse and (not pred_refuse):
        FN += 1 # 错误接受回答的样本数
        badcases.append({"id": qid, "type": "under_refuse", "question": g["question"], "best_score": p["best_score"]})
    else: # 正确接受回答
        TN += 1

    # retrieval 只评测可回答的
    if g["answerable"] is True:
        total_ans += 1
        exp_source = g.get("expected_source")
        exp_chunk = g.get("expected_chunk_id")

        retrieved = p.get("retrieved", [])[:k]

        def match_source(r):
            return (exp_source is not None) and (r.get("source") == exp_source)

        def match_chunk(r):
            return (exp_chunk is not None) and (r.get("chunk_id") == exp_chunk)

        if retrieved: # 如果检索到的文档来源或chunk不为空
            if match_source(retrieved[0]): # 如果检索到的文档来源与标准答案中的文档来源匹配
                hit_source_at1 += 1
            if match_chunk(retrieved[0]): # 如果检索到的文档chunk与标准答案中的文档chunk匹配
                hit_chunk_at1 += 1

        if any(match_source(r) for r in retrieved): # 如果检索到的文档来源中，有任何文档来源与标准答案中的文档来源匹配
            hit_source_atk += 1
        else:
            badcases.append({"id": qid, "type": "miss_source", "question": g["question"], "expected_source": exp_source})

        if any(match_chunk(r) for r in retrieved): # 如果检索到的文档chunk中，有任何文档chunk与标准答案中的文档chunk匹配
            hit_chunk_atk += 1
        else:
            badcases.append({"id": qid, "type": "miss_chunk", "question": g["question"], "expected_chunk_id": exp_chunk})

# metrics 计算命中率    
def safe_div(a, b):
    return a / b if b else 0.0 # 避免除0的情况

# ====== 收集指标和配置 ======
metrics = {
    "total_ans": total_ans,
    "hit_source@1": safe_div(hit_source_at1, total_ans),
    f"hit_source@{k}": safe_div(hit_source_atk, total_ans),
    "hit_chunk@1": safe_div(hit_chunk_at1, total_ans),
    f"hit_chunk@{k}": safe_div(hit_chunk_atk, total_ans),
    "TP": TP, "FP": FP, "FN": FN, "TN": TN,
    "precision": safe_div(TP, TP + FP),
    "recall": safe_div(TP, TP + FN),
}

# precision, recall, f1 计算
precision = metrics["precision"]
recall = metrics["recall"]
metrics["f1"] = safe_div(2 * precision * recall, precision + recall)

config = {
    "k": k,
    "threshold": 0.65,
    "model": "Qwen/Qwen2.5-32B-Instruct",
    "retrieval_mode": "vector",
    "gold_path": gold_path,
    "pred_path": pred_path,
    "notes": "baseline - 纯向量检索"
}

# ====== 使用 EvalLogger 记录日志 ======
logger = EvalLogger()
logger.log_config(config)
logger.log_metrics(metrics)
logger.save_predictions(pred_path)
logger.save_badcases(badcases)
logger.update_summary_csv(config, metrics)
logger.print_summary(metrics)

# ====== 仍然保留原始输出（控制台可见） ======
print("\n=== Retrieval (only answerable=True) ===")
print("answerable_total:", total_ans)
print("hit_source@1:", safe_div(hit_source_at1, total_ans))
print(f"hit_source@{k}:", safe_div(hit_source_atk, total_ans))
print("hit_chunk@1:", safe_div(hit_chunk_at1, total_ans))
print(f"hit_chunk@{k}:", safe_div(hit_chunk_atk, total_ans))

print("\n=== Refusal (all samples, positive=should_refuse) ===")
print("TP FP FN TN:", TP, FP, FN, TN)
print("precision:", precision)
print("recall:", recall)
print("f1:", metrics["f1"])

# 下载并保存错误的样本到badcases.jsonl文件中
with open("badcases.jsonl", "w", encoding="utf-8") as f:
    for x in badcases:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("\nWrote badcases.jsonl")

# 以上代码实现了对问答系统的评测，主要包括两个方面：1) 检索性能评测，计算检索到的文档来源和chunk的命中率；2) 拒答性能评测，计算模型在拒绝回答上的分类指标（precision、recall、f1）。同时还记录了错误的样本，方便后续分析和改进模型表现。同时还使用EvalLogger将评测结果记录到eval_logs/目录，支持多run历史对比。
