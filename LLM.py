#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

#定义rag的prompt模板，提示llm模型进行回答
RAG_PROMPT_TEMPLATE = """
你是一个严谨的制度/资料问答助手。你必须【只依据上下文证据】回答，不允许使用上下文之外的知识补充、猜测或发挥。请严格基于提供的参考资料作答。如果在参考资料中找不到明确答案，请不要自己编造，直接输出‘抱歉，资料库中未找到相关内容。

【强制规则】
1) 只能使用“上下文证据”中的信息作答；上下文没写到的内容，必须明确回答“未提及/未在资料中找到”，不要编造。
2) 回答必须包含引用：每条关键结论后面必须给出引用，引用格式为：
   - [chunk N]（N 必须来自上下文中的 chunk 编号）
   - 如果上下文里还包含 source/文件名，也可以在引用中附带，如 [chunk N | source=...]
3) 若上下文证据不足以回答问题，输出：
   - Answer: 未在资料中找到相关条款。
   - References: []

【用户问题】
{question}

【上下文证据】
{context}

【输出格式（必须严格遵守）】
Answer:
- 结论1 ... [chunk N]
- 结论2 ... [chunk M]

References: [chunk N, chunk M]
"""


class BaseModel:  #实现一个基类，定义chat和load_model方法，在不同的模型子类中实现
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):  #这里使用的是openai的chat模型，调用模型api
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:

        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_URL
        )

        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})  #将用户输入的问题和检索到的上下文按照prompt格式进行拼接，将其添加至模型历史中，在对话时可以发送给模型进行回答
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1 #控制模型生成的多样性，值越低生成的内容越保守和确定，值越高生成的内容越多样和有创意，根据实际需求调整这个参数
        )
        return response.choices[0].message.content #这里按照openai的api返回格式进行输出
