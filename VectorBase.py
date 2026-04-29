#!/usr/bin/env python
# -*- coding: utf-8 -*-  #规定该文件的编码格式为uft-8



import os
from typing import Dict, List, Optional, Tuple, Union
import json
from Embeddings import BaseEmbeddings, OpenAIEmbedding  #从embeddings模块中导入BaseEmbeddings和OpenAIEmbedding类，使用embedding模型进行文本向量化，这里的embedding模型相较于llm模型来说，功能更单一，专注于将文本转换为向量表示
import numpy as np
from tqdm import tqdm


class VectorStore:
    def __init__(self, document: List[dict] = ['']) -> None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        
        self.vectors = []
        texts=[i['text']  for i in self.document]
        for doc in tqdm(texts, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc)) #调用embedding模型的get_embedding方法，将文本转换为向量表示，并将结果保存在self.vectors列表中
        return self.vectors

    def persist(self, path: str = 'storage'):  #该函数的作用是将当前的文档和生成的向量库保存至本地
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):  #在第一次使用embedding模型生成之后，向量库和文档已经保存至本地，可以直接从本地加载向量库
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2) #调用BaseEmbeddings类中的cosine_similarity方法，计算两个向量之间的余弦相似度，返回一个介于-1和1之间的值，表示两个向量的相似程度

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> Tuple[List[str],List[int],List[str],List[float]]:  #该函数将用户输入的问题转化为向量表示，之后计算用户问题与向量库中的每个向量之间的相似度，并返回相似度最高的k条结果，为了便于分析，这里返回了source,chunk_id,text和score
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                          for vector in self.vectors])
        return np.array([i['source'] for i in self.document])[result.argsort()[-k:][::-1]].tolist(),np.array([i['chunk_id'] for i in self.document])[result.argsort()[-k:][::-1]].tolist(),np.array([i['text'] for i in self.document])[result.argsort()[-k:][::-1]].tolist(),np.sort(result)[::-1][:k].tolist()
