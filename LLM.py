#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

RAG_PROMPT_TEMPLATE = """
你是一个严谨的制度/资料问答助手。你必须【只依据上下文证据】回答，不允许使用上下文之外的知识补充、猜测或发挥。

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


class BaseModel:
    def __init__(self, model) -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass

class OpenAIChat(BaseModel):
    def __init__(self, model: str = "Qwen/Qwen2.5-32B-Instruct") -> None:
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        client = OpenAI(
            api_key='sk-qhjesqkxcwedutigqetlvsobnuwxgcguxyqjvpgscsxjtdxa',
            base_url='https://api.siliconflow.cn/v1'
        )

        history.append({'role': 'user', 'content': RAG_PROMPT_TEMPLATE.format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=2048,
            temperature=0.1
        )
        return response.choices[0].message.content
