from rank_bm25 import BM25Okapi
from VectorBase import VectorStore
import json


class HybridSearch:
    def __init__(self,vector_store,doc_path):
        self.vector_search=vector_store #保存的向量数据
        self.doc_path=doc_path #文本数据的路径
        self.docs=self._load_texts(doc_path) #加载文本数据，加载后docs是一个列表，每个元素是一个字典，包含source、chunk_id、text等字段
        tokenized_corpus=[self.docs[i]['text'] for i in range(len(self.docs))] #获取文本数据中的text字段，组成一个列表，每个元素是一个文本字符串
        self.bm25=BM25Okapi(tokenized_corpus) #进行分词处理，构建BM25模型


    @staticmethod
    def _load_texts(doc_path):
        with open(doc_path,'r',encoding='utf-8') as f:
            line=json.load(f) #f是一行json数据，line是一个列表，每个元素是一个字典，包含source、chunk_id、text等字段
            docs=[]
            for file in line:
                docs.append(file)
        return docs
    
        
    def query(self,query_text,Embedding_model,k=5):
        tokenized_query=list(query_text) #查询文本进行分词处理，得到一个列表，每个元素是一个词语
        bm25_scores=self.bm25.get_scores(tokenized_query)

        query_vector=Embedding_model.get_embedding(query_text)
        vector_scores=[]
        # ========== 错误1已修复：原来是 self.vector_search（对象），改为 self.vector_search.vectors（列表） ==========
        for vector in self.vector_search.vectors:
            score=self.vector_search.get_similarity(query_vector,vector) #计算查询向量与每个文本向量的余弦相似度，得到一个分数，分数越高表示文本与查询越相关
            vector_scores.append(score)
        
        #对两个分数取前三十，得到两个列表，每个列表包含30个元组，元组的第一个元素是文本的索引，第二个元素是分数
        bm25_top30=sorted(enumerate(bm25_scores),key=lambda x:x[1],reverse=True)[:30] #格式为[(index, score), ...]，按照分数从高到低排序，取前30个
        vector_top30=sorted(enumerate(vector_scores),key=lambda x:x[1],reverse=True)[:30]

        bm25_top30_list=[]
        for item in bm25_top30: #添加文本信息到bm25_top30_list列表中，每个元素是一个字典，包含source、chunk_id、score、content等字段
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
            
        rrf_scores={} #注意排名从1开始，所以在计算RRF分数时，分母是60加上排名，而不是60加上索引
        # ========== 错误2已修复：用 index（列表下标）作为 key，而不是 chunk_id ==========
        for rank, (index, score) in enumerate(bm25_top30, start=1):
            rrf_scores[index] = rrf_scores.get(index, 0) + 1 / (60 + rank) #BM25分数的排名越靠前，分数越高
        for rank, (index, score) in enumerate(vector_top30, start=1):
            rrf_scores[index] = rrf_scores.get(index, 0) + 1 / (60 + rank) #向量分数的排名越靠前，分数越高



        rrf_topk=sorted(rrf_scores.items(),key=lambda x:x[1],reverse=True)[:k] #字典重新排序，得到一个列表，每个元素是一个元组，元组的第一个元素是文本的索引，第二个元素是RRF分数
        sources=[]
        chunk_ids=[]
        contents=[]
        scores=[]
        # ========== 错误2已修复：item[0] 现在是列表下标，不是 chunk_id ==========
        for idx,score in rrf_topk:
            source=self.docs[idx]['source']
            chunk_id=self.docs[idx]['chunk_id']
            content=self.docs[idx]['text']
            sources.append(source)
            chunk_ids.append(chunk_id)
            contents.append(content)
            scores.append(score)
        return sources,chunk_ids,contents,scores,max(vector_scores) if vector_scores else 0.0 #返回向量余弦最高分，专门用于拒答判断


#经过实验，使用混合搜索的方式，在目前测试集上使得f1score提升至满分，precision和recall都达到1.0，说明混合搜索的方式能够有效地提升检索的准确率和召回率，从而提高模型的整体性能。通过结合BM25和向量搜索的优势，混合搜索能够更全面地捕捉文本之间的相关性，提供更准确的检索结果，进而提升问答系统的表现。