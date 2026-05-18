import numpy as np 
from FlagEmbedding  import FlagReranker
import json
from dotenv import load_dotenv
import os

load_dotenv() # 加载环境变量，确保在使用API接口时能够正确获取到API密钥和其他配置信息，这对于调用OpenAI的API接口进行问答和文本处理是必要的。
url=os.getenv("HF_ENDPOINT") #从环境变量中获取OpenAI API的基础URL，确保在调用API接口时能够正确连接到指定的服务器地址，这对于使用OpenAI的服务进行文本处理和问答是必要的。

class rerank:
    def __init__(self,model_name='BAAI/bge-reranker-v2-m3',use_fp16=True):
        # BUG1: 参数名错误——旧版写 model_name=，新版 API 参数名叫 model_name_or_path，写错会导致模型路径不进参数、加载失败
        self.reranker=FlagReranker(model_name_or_path=model_name,use_fp16=use_fp16)

    def rerank(self,query_text,candidate,topk=20):
        pairs=[]

        for item in candidate:
            pair=[query_text,item['content']]
            pairs.append(pair)

        # BUG2: API 变更——新版 FlagEmbedding 废弃 __call__，必须用 compute_score()；不加 normalize=True 返回原始 logits（可能负数），与阈值 0.65 不兼容
        scores=self.reranker.compute_score(pairs, normalize=True)
        for score,item in zip(scores,candidate):
            item['rerank_score']=score #将分数添加到候选文本的字典中，方便后续排序和筛选

        rerank=sorted(candidate,key=lambda x:x['rerank_score'],reverse=True) #对候选文本按照rerank_score进行排序，得到一个新的列表，包含topk个分数最高的候选文本
        return rerank[:topk]
    

        



    