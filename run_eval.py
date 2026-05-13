import json
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding
from LLM import OpenAIChat
from HybridSearch import HybridSearch

gold_path = "eval_draft1.jsonl"
hybrid_pred_path = "pred_hybrid.jsonl"

k = 5
threshold = 0.65
model_name = "Qwen/Qwen2.5-32B-Instruct"

vector = VectorStore()
embedding = OpenAIEmbedding()
vector.load_vector("./storage")

llm = OpenAIChat(model=model_name) # 初始化LLM

#初始化混合检索器，传入向量存储对象和文本数据路径
hybrid=HybridSearch(vector_store=vector, doc_path="./storage/doecment.json")

def build_context(sources, chunk_ids, contents, scores): #构建上下文，将检索到的文档内容和元信息格式化为上下文
    ctx = ""
    for j, chunk in enumerate(contents):
        ctx += (
            f"\n\n====chunk {j},score:{float(scores[j]):.4f},"
            f"source:{sources[j]},chunk_id:{chunk_ids[j]}====\n{chunk}\n"
        )
    return ctx

with open(gold_path, "r", encoding="utf-8") as fin, \
     open(hybrid_pred_path, "w", encoding="utf-8") as fout: # 打开输入文件和输出文件

    for line in fin:
        if not line.strip():
            continue
        gold = json.loads(line) #导入标准集数据

        qid = gold["id"]
        question = gold["question"]

        # ========== 混合检索：RRF 做排序，向量余弦最高分做拒答 ==========
        hybrid_sources, hybrid_chunk_ids, hybrid_contents, hybrid_scores, max_cos = hybrid.query(
            question, Embedding_model=embedding, k=k
        )
        best_score =float(max_cos)  # 用向量余弦最高分做拒答，和纯向量检索保持同一量纲
        refused = bool(best_score < threshold)

        retrieved = [] # 存储检索到的文档信息，包括来源、chunk_id、分数和内容
        for j in range(len(hybrid_contents)): #保存top_5的检索信息，便于之后的评估分析
            retrieved.append({
                "source": hybrid_sources[j],
                "chunk_id": int(hybrid_chunk_ids[j]) if hybrid_chunk_ids[j] is not None else None,
                "score": float(hybrid_scores[j]),
                "content": hybrid_contents[j],
            })

        if refused: # 如果检索到的文档分数低于阈值，拒绝回答，这是第一层的拒答逻辑
            answer = ""
        else:
            context = build_context(hybrid_sources, hybrid_chunk_ids, hybrid_contents, hybrid_scores)
            answer = llm.chat(question, [], context) # 调用LLM生成回答

        pred = {
            "id": qid,
            "question": question,
            "refused": refused,
            "best_score": best_score,
            "retrieved": retrieved,
            "answer": answer,
        }

        fout.write(json.dumps(pred, ensure_ascii=False) + "\n") # 写入预测结果到文件

print(f"Done. Wrote {hybrid_pred_path}")
