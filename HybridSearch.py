import rank_bm25
from VectorBase import VectorBase


class HybridSearch:
    def __init__(self,vector_store,doc_path):
        self.vector_search=vector_store
        self.doc_path=doc_path
        self.docs=self._load_texts()
        

    @staticmethod
    def _load_texts(doc_path):
        with open(doc_path,'r',encoding='utf-8') as f:
            line=f.readlines()
            docs=[]
            for file in line:
                docs.append({
                    'source':file['source'],
                    'text':file['text'],
                })
        return docs
    
    def build_bm25(self):
        tokenized_corpus=[self.docs[i]['text'] for i in range(len(self.docs))]
        self.bm25=rank_bm25.BM250api(tokenized_corpus)

    def query(self,query_text,Embedding_model,k=5):
        tokenized_query=list(query_text)
        bm25_scores=self.bm25.get_scores(tokenized_query)

        query_vector=Embedding_model.get_embedding(query_text)
        vector_scores=[]
        for vector in self.vector_store:
            score=VectorBase.cosine_similarity(query_vector,vector)
            vector_scores.append(score)
        
        bm25_top30=sorted(enumerate(bm25_scores),key=lambda x:x[1],reverse=True)[:30]
        vector_top30=sorted(enumerate(vector_scores),key=lambda x:x[1],reverse=True)[:30]

        

        





