import json
from VectorBase import VectorStore
from Embeddings import OpenAIEmbedding

vector = VectorStore()
embedding = OpenAIEmbedding()
vector.load_vector('./storage')

threshold = 0.35

in_path = './eval_top10.jsonl'
out_path = './eval_autolabeled.jsonl'  # 不建议覆盖原文件

with open(in_path, 'r', encoding='utf-8') as fin, \
     open(out_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        if not line.strip():
            continue
        item = json.loads(line)

        question = item['question']
        source, chunk_id, content, scores = vector.query(
            question, EmbeddingModel=embedding, k=5
        )

        best_score = float(scores[0]) if scores else 0.0

        # 自动打标（注意是 bool）
        item['answerable'] = (best_score >= threshold)

        # 可选：把一些关键信息也记录下来，方便你人工复核
        item['auto_best_score'] = best_score
        item['auto_top1_source'] = source[0] if source else None
        item['auto_top1_chunk_id'] = int(chunk_id[0]) if chunk_id else None

        fout.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Done. Wrote: {out_path}")