import json

gold_path = "eval_draft.jsonl"
pred_path = "pred.jsonl"
k = 5

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

gold = {x["id"]: x for x in load_jsonl(gold_path)}
pred = {x["id"]: x for x in load_jsonl(pred_path)}

# ====== 1) Retrieval: source/chunk 命中率（只对 answerable=True 的样本） ======
total_ans = 0
hit_source_at1 = 0
hit_source_atk = 0
hit_chunk_at1 = 0
hit_chunk_atk = 0

# ====== 2) Refusal: 拒答分类指标（对所有样本） ======
TP = FP = FN = TN = 0  # 这里的“正类”定义为 should_refuse=True

badcases = []

for qid, g in gold.items():
    p = pred.get(qid)
    if p is None:
        badcases.append({"id": qid, "type": "missing_pred"})
        continue

    should_refuse = (g["answerable"] is False)
    pred_refuse = (p["refused"] is True)

    if should_refuse and pred_refuse:
        TP += 1
    elif (not should_refuse) and pred_refuse:
        FP += 1
        badcases.append({"id": qid, "type": "over_refuse", "question": g["question"], "best_score": p["best_score"]})
    elif should_refuse and (not pred_refuse):
        FN += 1
        badcases.append({"id": qid, "type": "under_refuse", "question": g["question"], "best_score": p["best_score"]})
    else:
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

        if retrieved:
            if match_source(retrieved[0]):
                hit_source_at1 += 1
            if match_chunk(retrieved[0]):
                hit_chunk_at1 += 1

        if any(match_source(r) for r in retrieved):
            hit_source_atk += 1
        else:
            badcases.append({"id": qid, "type": "miss_source", "question": g["question"], "expected_source": exp_source})

        if any(match_chunk(r) for r in retrieved):
            hit_chunk_atk += 1
        else:
            badcases.append({"id": qid, "type": "miss_chunk", "question": g["question"], "expected_chunk_id": exp_chunk})

# metrics
def safe_div(a, b):
    return a / b if b else 0.0

print("=== Retrieval (only answerable=True) ===")
print("answerable_total:", total_ans)
print("hit_source@1:", safe_div(hit_source_at1, total_ans))
print(f"hit_source@{k}:", safe_div(hit_source_atk, total_ans))
print("hit_chunk@1:", safe_div(hit_chunk_at1, total_ans))
print(f"hit_chunk@{k}:", safe_div(hit_chunk_atk, total_ans))

print("\n=== Refusal (all samples, positive=should_refuse) ===")
precision = safe_div(TP, TP + FP)
recall = safe_div(TP, TP + FN)
f1 = safe_div(2 * precision * recall, precision + recall)
print("TP FP FN TN:", TP, FP, FN, TN)
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)

# dump badcases
with open("badcases.jsonl", "w", encoding="utf-8") as f:
    for x in badcases:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print("\nWrote badcases.jsonl")