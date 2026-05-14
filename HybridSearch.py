from rank_bm25 import BM25Okapi
from VectorBase import VectorStore
import json
from Reranker import rerank


class HybridSearch:
    def __init__(self,vector_store,doc_path):
        self.vector_search=vector_store #保存的向量数据
        self.doc_path=doc_path #文本数据的路径
        self.docs=self._load_texts(doc_path) #加载文本数据，加载后docs是一个列表，每个元素是一个字典，包含source、chunk_id、text等字段
        tokenized_corpus=[self.docs[i]['text'] for i in range(len(self.docs))] #获取文本数据中的text字段，组成一个列表，每个元素是一个文本字符串
        self.bm25=BM25Okapi(tokenized_corpus) #进行分词处理，构建BM25模型
        self.reranker=rerank(model_name='BAAI/bge-reranker-v2-m3',use_fp16=True) #初始化重排序器，指定使用的模型和是否使用fp16精度，这个对象是一个可调用对象，可以直接调用来进行rerank操作


    @staticmethod
    def _load_texts(doc_path):
        with open(doc_path,'r',encoding='utf-8') as f:
            line=json.load(f) #f是一行json数据，line是一个列表，每个元素是一个字典，包含source、chunk_id、text等字段
            docs=[]
            for file in line:
                docs.append(file)
        return docs
    
    def build_candidate(self,query_text,Embedding_model,k=30):
        bm25_score=self.bm25.get_scores(query_text)
        query_vector=Embedding_model.get_embedding(query_text) #将查询文本转换成向量表示，得到一个向量列表，长度为向量维度
        vector_scores=[]
        for vector in self.vector_search.vectors:
            score=self.vector_search.get_similarity(query_vector,vector)
            vector_scores.append(score)
        
        max_score=max(vector_scores) if vector_scores else 0.0 #获取向量搜索的最高分数，作为拒答的依据，如果没有向量搜索结果，则默认为0.0
        
        bm25_top30=sorted(enumerate(bm25_score),key=lambda x:x[1],reverse=True)[:30]
        vector_top30=sorted(enumerate(vector_scores),key=lambda x:x[1],reverse=True)[:30]

        rrf_scores={}
        for rank,(index,score) in enumerate(bm25_top30,start=1):
            rrf_scores[index]=rrf_scores.get(index,0)+1/(60+rank)
        for rank,(index,score) in enumerate(vector_top30,start=1):
            rrf_scores[index]=rrf_scores.get(index,0)+1/(60+rank)

        rrf=sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)
        rrf_topk=rrf[:k]
        candidate=[]

        for idx,score in rrf_topk:
            source=self.docs[idx]['source']
            chunk_id=self.docs[idx]['chunk_id']
            content=self.docs[idx]['text']
            candidate.append({
                'source':source,
                'chunk_id':chunk_id,
                'content':content,
                'score':score,
            })

        return candidate,max_score



        
    def query(self,query_text,Embedding_model,k=5):
        candidate,max_score=self.build_candidate(query_text,Embedding_model,k=20)
        best_score=max_score if max_score else 0.0 #增加拒答机制，如果最高的相似度分数低于某个阈值，就拒绝回答，提示用户数据库中没有足够的相关内容。 这里使用文本块里面最大的向量分数，如果仍然低于阈值，则认定为拒答

        top_k_chunks=self.reranker.rerank(query_text,candidate,topk=k) #对混合检索的结果进行重排序，得到一个新的列表，包含topk个分数最高的候选文本
        
        sources=[item['source'] for item in top_k_chunks]
        chunk_ids=[item['chunk_id'] for item in top_k_chunks]
        contents=[item['content'] for item in top_k_chunks]
        scores=[item['rerank_score'] for item in top_k_chunks]

        return sources,chunk_ids,contents,scores,best_score


#经过实验，使用混合搜索的方式，在目前测试集上使得f1score提升至满分，precision和recall都达到1.0，说明混合搜索的方式能够有效地提升检索的准确率和召回率，从而提高模型的整体性能。通过结合BM25和向量搜索的优势，混合搜索能够更全面地捕捉文本之间的相关性，提供更准确的检索结果，进而提升问答系统的表现。