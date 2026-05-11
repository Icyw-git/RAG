import rank_bm25
from VectorBase import VectorBase
import json


class HybridSearch:
    def __init__(self,vector_store,doc_path):
        self.vector_search=vector_store #保存的向量数据
        self.doc_path=doc_path #文本数据的路径
        self.docs=self._load_texts(doc_path) #加载文本数据，加载后docs是一个列表，每个元素是一个字典，包含source、chunk_id、text等字段
        

    @staticmethod
    def _load_texts(doc_path):
        with open(doc_path,'r',encoding='utf-8') as f:
            line=json.load(f)
            docs=[]
            for file in line:
                docs.append(file)
        return docs
    
    def build_bm25(self):
        tokenized_corpus=[self.docs[i]['text'] for i in range(len(self.docs))]
        self.bm25=rank_bm25.BM25Okapi(tokenized_corpus)

    def query(self,query_text,Embedding_model,k=5):
        tokenized_query=list(query_text)
        bm25_scores=self.bm25.get_scores(tokenized_query)

        query_vector=Embedding_model.get_embedding(query_text)
        vector_scores=[]
        for vector in self.vector_search:
            score=VectorBase.cosine_similarity(query_vector,vector)
            vector_scores.append(score)
        
        bm25_top30=sorted(enumerate(bm25_scores),key=lambda x:x[1],reverse=True)[:30]
        vector_top30=sorted(enumerate(vector_scores),key=lambda x:x[1],reverse=True)[:30]

        bm25_top30_list=[]
        for item in bm25_top30:
            bm25_top30_list.append({
                'source':self.docs[item[0]]['source'],
                'chunk_id':self.docs[item[0]]['chunk_id'],
                'score':item[1],
                'content':self.docs[item[0]]['text'],
            })

        vector_top30_list=[]
        for item in vector_top30:
            vector_top30_list.append({
                'source':self.docs[item[0]]['source'],
                'chunk_id':self.docs[item[0]]['chunk_id'],
                'score':item[1],
                'content':self.docs[item[0]]['text'],
            })
            
        rrf_scores={}
        bm25_id_rank={item['chunk_id']:i+1 for i,item in enumerate(bm25_top30_list)}
        vector_id_rank={item['chunk_id']:i+1 for i,item in enumerate(vector_top30_list)}
        for item in bm25_top30_list:
            rrf_scores[item['chunk_id']]=rrf_scores.get(item['chunk_id'],0)+1/(60+bm25_id_rank[item['chunk_id']])
        for item in vector_top30_list:
            rrf_scores[item['chunk_id']]=rrf_scores.get(item['chunk_id'],0)+1/(60+vector_id_rank[item['chunk_id']])

        rrf_topk=sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)[:k]
        sources=[]
        chunk_ids=[]
        contents=[]
        scores=[]
        for item in rrf_topk:
            source=self.docs[item[0]]['source']
            chunk_id=item[0]
            content=self.docs[item[0]]['text']
            score=item[1]
            sources.append(source)
            chunk_ids.append(chunk_id)
            contents.append(content)
            scores.append(score)
        return sources,chunk_ids,contents,scores
    


            



        





