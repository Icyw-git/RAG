import json
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding

questions=[]
with open('eval.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line) # 解析每行的JSON数据
            new_question={
                "id": data["id"],
                "question":data['question'],
                'answerable':data['answerable'],

            } #创建一个新的字典，只保留id和question字段
            questions.append(new_question)

print(questions)



k = 10
out = []

vector=VectorStore() # 创建一个新的VectorStore对象
embedding=OpenAIEmbedding()
vector.load_vector('./storage') # 加载本地的数据库

for q in questions:
    sources, chunk_ids, texts, scores = vector.query(q["question"], EmbeddingModel=embedding, k=k)
    # 草稿：先取 top1 作为建议证据
    out.append({
        "id": q["id"],
        "question": q["question"],
        "answerable": q['answerable'],
        "expected_source": sources[0] if q['answerable'] else None, #如果不是answerable的样本就不填source和chunk_id，避免误导模型
        "expected_chunk_id": chunk_ids[0] if q['answerable'] else None,
        "note": "AUTO_DRAFT: please verify by reading top-k" if q['answerable'] else 'AUTO_DRAFT: out of KB, should refuse',
    })

with open("eval_draft1.jsonl","w",encoding="utf-8") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n") # 将每个字典对象转换为JSON字符串，并写入文件，每行一个JSON对象


#这个脚本的作用是获取每个样本的top_1检索结果，作为模型回答的建议证据，并标注出哪些样本应该拒答。通过分析这些建议证据和拒答样本，可以评估模型的检索能力和拒答策略的合理性，为后续优化提供参考依据。