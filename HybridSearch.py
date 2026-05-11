import rank_bm25
from VectorBase import VectorBase
import json


class HybridSearch:
    def __init__(self,vector_store,doc_path):
        self.vector_search=vector_store
        self.doc_path=doc_path
        self.docs=self._load_texts(doc_path)
        

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
            
            



        





