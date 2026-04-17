import json
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding

questions=[]
with open('eval.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        new_question={
            "id": data["id"],
            "question":data['question']
        }
        questions.append(new_question)

print(questions)

vector=VectorStore() # 创建一个新的VectorStore对象
embedding=OpenAIEmbedding()
vector.load_vector('./storage') # 加载本地的数据库

k=10
top10=[]
for i in questions:

    source,chunk_id,content,scores = vector.query(i['question'], EmbeddingModel=embedding, k=k)
    top10.append({
        "id": i['id'],
        "question": i['question'],
        "top10": [
            {
                "source": source[j],
                "chunk_id": chunk_id[j],
                "content": content[j],
                "score": scores[j]
            }
            for j in range(k)
        ]
    }) # 将每个问题的前10条检索结果保存到一个新的字典中，包含问题的id、问题文本和一个列表，列表中每个元素是一个字典，包含source、chunk_id、content和score字段

with open("eval_top10.jsonl","w",encoding="utf-8") as f:
    for row in top10:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")