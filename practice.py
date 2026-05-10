"""
=========================================================
  练习文件：从各个模块提取的核心代码训练
=========================================================
  
  题目来源于项目中以下模块的核心逻辑：
    Embeddings.py   → 余弦相似度（纯 Python）
    VectorBase.py   → Top-K 排序、列表推导取字段
    LLM.py          → prompt 模板格式化
    utils.py        → token 计算
    scores.py       → 混淆矩阵、safe_div、load_jsonl
    run_eval.py     → 构建上下文
    eval_logger.py  → 嵌套字典展平、时间戳、CSV 追加
    demo.py         → 字符串格式化

使用方法：
  1. 将每个函数中的 "..." 替换为正确的代码
  2. 运行 python practice.py 验证答案
  3. 翻到底部查看参考答案
"""

import json
import datetime


# =========================================================
# 模块：Embeddings.py — 余弦相似度（★★☆☆☆）
# =========================================================

def cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度（纯 Python，不用 numpy）
    公式：dot(v1,v2) / (|v1| * |v2|)
    输入: v1, v2 是等长的 float 列表
    输出: float，范围 [-1, 1]

    填空示例：
      点积 = sum(a * b for a, b in zip(v1, v2)) #zip函数的作用是将两个列表的元素一一对应打包成元组，方便同时遍历
      范数 = sum(x * x for x in v) ** 0.5
    """
    # 计算点积
    dot_product =sum(a*b  for a,b in zip(v1,v2) )
      # 用 sum + zip
    
    # 计算两个向量的长度
    norm_v1 = pow(sum(a*a for a in v1),0.5)

          # 用 sum + **0.5
    norm_v2 = pow(sum(b*b for b in v2),0.5)
    
    # 处理分母为 0
    if norm_v1==0  or norm_v2==0:
        return 0.0
    
    # 返回余弦相似度
    return dot_product/(norm_v1*norm_v2)



# =========================================================
# 模块：VectorBase.py — Top-K 排序（★★☆☆☆）
# =========================================================

def top_k_indices(scores, k):
    """
    返回分数最高的 k 个元素的索引（从高到低）
    示例：scores=[0.1, 0.8, 0.5, 0.9, 0.3], k=3 → [3, 1, 2]
    提示：
      1. enumerate(scores) 绑定索引和分数 → [(0,0.1), (1,0.8), ...]
      2. sort(key=lambda x: x[1], reverse=True) 按分数排序
      3. 取前 k 个的索引
    """
    # 你的代码写在这里
    sorted_scores=sorted(enumerate(scores),key=lambda x:x[1],reverse=True) #这里返回的是一个元组，第一个元素是索引，第二个元素是分数
    for i in range(k):
        sorted_scores[i]=sorted_scores[i][0]

    return sorted_scores[:k]




# =========================================================
# 模块：VectorBase.py — 列表推导式取字段（★★☆☆☆）
# =========================================================

def extract_fields(documents, indices, field):
    """
    从 documents 列表中提取指定索引的 field 字段
    示例：
      docs = [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}]
      extract_fields(docs, [1, 0], "text") → ["b", "a"]
    提示：[documents[i][field] for i in indices]
    """
    # 你的代码写在这里
    fields=[]
    for i in indices:
        fields.append(documents[i][field])

    return fields


# =========================================================
# 模块：scores.py — 混淆矩阵（★★★☆☆）
# =========================================================

def confusion_matrix(y_true, y_pred):
    """
    计算二分类的 TP, FP, FN, TN
    输入: y_true, y_pred 是两个等长的 bool 列表
    输出: (TP, FP, FN, TN)
    """
    TP = FP = FN = TN = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP+=1
        elif not true and pred:
            FP+=1
        elif true and not pred:
            FN+=1
        else:
            TN+=1
    return TP, FP, FN, TN


# =========================================================
# 模块：scores.py — safe_div（★★☆☆☆）
# =========================================================

def safe_div(a, b):
    """安全的除法，分母为 0 时返回 0.0，一行代码"""
    return 0.0 if b==0 else a/b # 使用安全的除法


# =========================================================
# 模块：scores.py — precision/recall/f1（★★★☆☆）
# =========================================================

def calc_metrics(TP, FP, FN, TN):
    """
    根据 TP/FP/FN/TN 计算 precision, recall, f1
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1        = 2 * precision * recall / (precision + recall)
    所有除法都使用 safe_div
    """

    precision = safe_div(TP,TP+FP)
    recall = safe_div(TP,TP+FN) #召回率的分母是 TP + FN，表示所有实际为正的样本数
    f1 = safe_div(2*precision*recall, precision+recall)
    return precision, recall, f1


# =========================================================
# 模块：eval_logger.py — 嵌套字典展平（★★★★☆）
# =========================================================

def flatten_dict(d, prefix=""):
    """
    将嵌套字典展平为单层
    示例：
      flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
      → {"a_b": 1, "a_c": 2, "d": 3}
    
      flatten_dict({"a": {"b": 1}}, prefix="cfg")
      → {"cfg_a_b": 1}
    
    逻辑：
      for key, val in d.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(val, dict):
          递归调用 flatten_dict(val, prefix=flat_key)
        else:
          flat[flat_key] = val
    """
    flat = {}
    for key, val in d.items():
        flat_key = f"{prefix}_{key}" if prefix else key
        if isinstance(val, dict):
            flat.update(flatten_dict(val,prefix=flat_key))
            
        else:
            flat[flat_key]=val
    return flat


# =========================================================
# 模块：eval_logger.py — 时间戳生成（★☆☆☆☆）
# =========================================================

def generate_run_id():
    """生成运行 ID，格式：YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# 模块：eval_logger.py — CSV 行拼接（★★☆☆☆）
# =========================================================

def dict_to_csv_row(d):
    """
    将字典转成 CSV 的一行（逗号分隔）
    示例：{"a": 1, "b": "hello"}
      表头 → "a,b"
      数据 → "1,hello"
    返回 (header, data_row)
    """
    header = ','.join(d.keys())
    data_row = ','.join(str(i) for i in d.values())
    return header, data_row


# =========================================================
# 模块：LLM.py — prompt 模板格式化（★★☆☆☆）
# =========================================================

def format_prompt(question, context):
    """用模板填充问题和上下文"""
    template = """用户问题：{question}
参考材料：{context}
请基于以上材料回答问题。"""
    return template.format(question=question,context=context)


# =========================================================
# 模块：run_eval.py — 构建上下文（★★☆☆☆）
# =========================================================

def build_context(sources, chunk_ids, contents, scores):
    """
    将检索结果拼接成带格式的上下文字符串
    每条格式：
      ====chunk {j},score:{float(scores[j]):.4f},source:{sources[j]},chunk_id:{chunk_ids[j]}====
      {contents[j]}
    """
    ctx = ""
    for j in range(len(contents)):
        ctx += f'====chunk {j},score:{float(scores[j]):.4f},source:{sources[j]},chunk_id:{chunk_ids[j]}====\n{contents[j]}\n'  # 拼接每条 chunk
    return ctx


# =========================================================
# 模块：demo.py — 字符串格式化（★☆☆☆☆）
# =========================================================

def format_score(score):
    """将分数格式化为保留 4 位小数的字符串，如 0.965517 → "0.9655" """
    return f'{score:.4f}'


# =========================================================
# 模块：utils.py — 估算 token 数（★★☆☆☆）
# =========================================================

def estimate_tokens(text):
    """
    估算文本的 token 数量
    简化规则：
      - 英文字母/数字: 每个 0.25 个 token
      - 中文字符 (ord > 127): 每个 2 个 token
      - 其他: 每个 0.5 个 token
    """
    tokens = 0
    for char in text:
        if 'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9':
            tokens += 0.25
        elif ord(char) > 127:
            tokens += 2
        else:
            tokens += 0.5
    return tokens


# =========================================================
# 测试代码
# =========================================================

def run_tests():
    passed = 0
    total = 0

    # 1: 余弦相似度
    total += 1
    try:
        v1, v2 = [1, 2, 3], [4, 5, 6]
        result = cosine_similarity(v1, v2)
        expected = 0.9746318461970762
        assert abs(result - expected) < 1e-6, f"期望 {expected:.4f}, 实际 {result:.4f}"
        print("✅ 练习 1 通过（余弦相似度）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 1 失败: {e}")

    # 2: Top-K 排序
    total += 1
    try:
        result = top_k_indices([0.1, 0.8, 0.5, 0.9, 0.3], 3)
        assert result == [3, 1, 2], f"期望 [3, 1, 2], 实际 {result}"
        print("✅ 练习 2 通过（Top-K 排序）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 2 失败: {e}")

    # 3: 列表推导式
    total += 1
    try:
        docs = [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}, {"id": 2, "text": "c"}]
        result = extract_fields(docs, [2, 0], "text")
        assert result == ["c", "a"], f"期望 ['c', 'a'], 实际 {result}"
        print("✅ 练习 3 通过（列表推导式）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 3 失败: {e}")

    # 4: 混淆矩阵
    total += 1
    try:
        y_true = [True, False, True, False, True]
        y_pred = [True, True, False, False, True]
        TP, FP, FN, TN = confusion_matrix(y_true, y_pred)
        assert (TP, FP, FN, TN) == (2, 1, 1, 1), f"期望 (2,1,1,1), 实际 ({TP},{FP},{FN},{TN})"
        print("✅ 练习 4 通过（混淆矩阵）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 4 失败: {e}")

    # 5: safe_div
    total += 1
    try:
        assert safe_div(10, 2) == 5.0
        assert safe_div(10, 0) == 0.0
        print("✅ 练习 5 通过（safe_div）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 5 失败: {e}")

    # 6: PRF
    total += 1
    try:
        p, r, f = calc_metrics(50, 10, 5, 100)
        assert abs(p - 0.8333) < 0.01
        assert abs(r - 0.9091) < 0.01
        assert abs(f - 0.8696) < 0.01
        print("✅ 练习 6 通过（PRF 计算）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 6 失败: {e}")

    # 7: 嵌套字典展平
    total += 1
    try:
        result = flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        expected = {"a_b": 1, "a_c": 2, "d": 3}
        assert result == expected, f"期望 {expected}, 实际 {result}"
        
        result2 = flatten_dict({"a": {"b": 1}}, prefix="cfg")
        expected2 = {"cfg_a_b": 1}
        assert result2 == expected2
        print("✅ 练习 7 通过（嵌套字典展平）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 7 失败: {e}")

    # 8: 时间戳
    total += 1
    try:
        run_id = generate_run_id()
        assert len(run_id) == 15 and "_" in run_id
        print(f"✅ 练习 8 通过（时间戳: {run_id}）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 8 失败: {e}")

    # 9: CSV
    total += 1
    try:
        header, row = dict_to_csv_row({"a": 1, "b": "hello"})
        assert header == "a,b", f"期望 'a,b', 实际 '{header}'"
        assert row == "1,hello", f"期望 '1,hello', 实际 '{row}'"
        print("✅ 练习 9 通过（CSV 行拼接）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 9 失败: {e}")

    # 10: prompt
    total += 1
    try:
        result = format_prompt("加分政策是什么", "文件规定...")
        assert "加分政策是什么" in result
        assert "文件规定..." in result
        print("✅ 练习 10 通过（prompt 格式化）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 10 失败: {e}")

    # 11: 构建上下文
    total += 1
    try:
        ctx = build_context(["a.pdf"], [0], ["内容是..."], [0.9567])
        assert "chunk 0" in ctx
        assert "a.pdf" in ctx
        assert "0.9567" in ctx
        print("✅ 练习 11 通过（构建上下文）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 11 失败: {e}")

    # 12: 分数格式化
    total += 1
    try:
        result = format_score(0.965517)
        assert result == "0.9655", f"期望 '0.9655', 实际 '{result}'"
        print("✅ 练习 12 通过（分数格式化）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 12 失败: {e}")

    # 总结
    print(f"\n{'='*50}")
    print(f"结果: {passed}/{total} 通过")
    if passed == total:
        print("🎉 全部通过！")
    else:
        print(f"💪 {total - passed} 个未通过")


# =========================================================
# 参考答案
# =========================================================

"""
练习 1（余弦相似度）：
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

练习 2（Top-K 排序）：
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]

练习 3（列表推导式）：
    return [documents[i][field] for i in indices]

练习 4（混淆矩阵）：
    if true and pred:        TP += 1
    elif not true and pred:  FP += 1
    elif true and not pred:  FN += 1
    else:                    TN += 1

练习 5（safe_div）：
    return a / b if b else 0.0

练习 6（PRF）：
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

练习 7（嵌套字典展平）：
    flat.update(flatten_dict(val, prefix=flat_key))
    或者 flat[flat_key] = val

练习 8（时间戳）：
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

练习 9（CSV）：
    header = ",".join(d.keys())
    data_row = ",".join(str(v) for v in d.values())
    return header, data_row

练习 10（prompt）：
    return template.format(question=question, context=context)

练习 11（构建上下文）：
    ctx += f"\n\n====chunk {j},score:{float(scores[j]):.4f},source:{sources[j]},chunk_id:{chunk_ids[j]}====\n{contents[j]}\n"

练习 12（分数格式化）：
    return f"{score:.4f}"
"""

if __name__ == "__main__":
    run_tests()