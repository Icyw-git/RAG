import json
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding
from LLM import OpenAIChat

gold_path = "eval_draft1.jsonl"
pred_path = "pred1.jsonl"

k = 5
threshold = 0.65
model_name = "Qwen/Qwen2.5-32B-Instruct"

vector = VectorStore()
embedding = OpenAIEmbedding()
vector.load_vector("./storage")

llm = OpenAIChat(model=model_name) # 初始化LLM

def build_context(sources, chunk_ids, contents, scores): #构建上下文，将检索到的文档内容和元信息格式化为上下文
    ctx = ""
    for j, chunk in enumerate(contents):
        ctx += (
            f"\n\n====chunk {j},score:{float(scores[j]):.4f},"
            f"source:{sources[j]},chunk_id:{chunk_ids[j]}====\n{chunk}\n"
        )
    return ctx

with open(gold_path, "r", encoding="utf-8") as fin, \
     open(pred_path, "w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue
        gold = json.loads(line) #导入标准集数据

        qid = gold["id"]
        question = gold["question"]

        sources, chunk_ids, contents, scores = vector.query(
            question, EmbeddingModel=embedding, k=k
        )

        best_score = float(scores[0]) if scores else 0.0
        refused = best_score < threshold  #这里根据设定的阈值进行判断是否拒答

        retrieved = [] # 存储检索到的文档信息，包括来源、chunk_id、分数和内容
        for j in range(len(contents)): #保存top_5的检索信息，便于之后的评估分析
            retrieved.append({
                "source": sources[j],
                "chunk_id": int(chunk_ids[j]) if chunk_ids[j] is not None else None,
                "score": float(scores[j]),
                "content": contents[j],
            })

        if refused: # 如果检索到的文档分数低于阈值，拒绝回答，这是第一层的拒答逻辑，后续可以根据llm的回答进行第二层拒答
            answer = ""
        else:
            context = build_context(sources, chunk_ids, contents, scores)
            answer = llm.chat(question, [], context) # 调用LLM生成回答，使用前面构建的context作为模型输入

        pred = {
            "id": qid,
            "question": question,
            "refused": refused,
            "best_score": best_score,
            "retrieved": retrieved,
            "answer": answer,
        }
        fout.write(json.dumps(pred, ensure_ascii=False) + "\n") # 写入预测结果到文件，每个样本占一行

print(f"Done. Wrote {pred_path}") # 打印完成信息