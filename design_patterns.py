"""
=============================================================================
  设计模式学习全攻略：从入门到实战
  =============================================================================
  
  本文件涵盖 Python 中最常用的 12 种设计模式，分为三大类：
    🏗️ 创建型模式（5 种）：控制对象的创建过程
    🧱 结构型模式（4 种）：处理类与对象的组合
    🧠 行为型模式（3 种）：管理对象之间的交互与职责分配

  每个模式包含四个部分：
    📖 概念讲解 — 是什么、解决什么问题、何时使用
    👀 完整示例 — 可运行的代码演示核心思想
    ✏️ 填空练习 — 关键代码留空，动手补全
    🚀 RAG 实战 — 结合本项目的真实应用场景

  学习路线（建议 6 天）：
    Day 1：单例模式 + 简单工厂 + 工厂方法
    Day 2：抽象工厂 + 建造者模式
    Day 3：装饰器模式 + 适配器模式
    Day 4：代理模式 + 外观模式
    Day 5：策略模式 + 观察者模式
    Day 6：模板方法模式 + 综合复习

  使用方法：
    python design_patterns.py    → 运行所有练习测试

  =============================================================================
"""

import time
import json
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, List, Any, Callable, Optional, Type

# =============================================================================
# 序言：什么是设计模式？
# =============================================================================

"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🟢 设计模式是什么？

  设计模式是软件开发中反复出现的问题的经典解决方案。
  它们是前人在大量项目中总结出的"最佳实践"，就像编程领域的"兵法"。

  打个比方：
    - 算法        = 菜谱（具体步骤）
    - 设计模式    = 厨房布局（整体架构思路）

🟢 为什么要学设计模式？

  1. 提高代码可读性 — "哦，这里用了策略模式"比读 200 行业务逻辑快得多
  2. 提高可维护性 — 模式自带解耦，改一处不影响全局
  3. 团队协作 — 统一的"设计语言"，减少沟通成本
  4. 面试必备 — 中高级岗位几乎必考

🟢 设计模式三大分类：

  ┌─────────────┬──────────────────────────┬──────────────────────┐
  │   创建型     │  控制对象的创建过程        │  单例、工厂、建造者   │
  ├─────────────┼──────────────────────────┼──────────────────────┤
  │   结构型     │  处理类与对象的组合方式    │  装饰器、适配器、代理  │
  ├─────────────┼──────────────────────────┼──────────────────────┤
  │   行为型     │  管理对象间的交互与职责    │  策略、观察者、模板    │
  └─────────────┴──────────────────────────┴──────────────────────┘

🟢 Python 中的设计模式

  Python 的"动态类型 + 一等函数 + 装饰器语法"让很多经典模式实现起来
  比其他语言（如 Java）简洁得多。本教程会展示"Pythonic"的写法，
  让你用最少的代码获得最大的收益。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# =============================================================================
# 🏗️ 第一部分：创建型模式
# =============================================================================

# =============================================================================
# 1. 单例模式（Singleton Pattern）
# =============================================================================

print("=" * 60)
print("🏗️ 第一部分：创建型模式")
print("=" * 60)

print("\n" + "─" * 60)
print("模式 1：单例模式（Singleton Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：如何确保一个类在整个程序中只有一个实例？

场景：
  • 数据库连接池 — 不需要创建多个连接池
  • 配置管理器   — 全局配置应该只有一份
  • 日志记录器   — 所有模块往同一个日志写
  • 模型缓存     — Embedding 模型加载一次就够了

核心思想：无论调用多少次「创建」，始终返回同一个对象。

四种 Python 实现方式（由简到繁）：
  ① 模块单例（最简单，推荐日常使用）
  ② __new__ 方法（经典 OOP 方式）
  ③ 装饰器（Pythonic 方式）
  ④ 元类（最强大，适合框架级代码）
"""

# ── 实现 ①：模块单例 ──
print("\n  📌 实现 ①：模块单例（最简单）")
print("  " + "-" * 40)

# Python 模块天然是单例 —— import 只会执行一次模块代码
# 下面模拟一个「配置管理器」模块

_module_config = {
    "model_name": "qwen3:0.5b",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_k": 5,
}


def get_config(key, default=None):
    """获取配置项（线程安全，因为有 GIL）"""
    return _module_config.get(key, default)


def set_config(key, value):
    """设置配置项"""
    _module_config[key] = value


def get_all_config():
    """获取所有配置的副本（防止外部修改内部数据）"""
    return dict(_module_config)


# 测试：无论在多少个文件里 import，_module_config 始终是同一个字典
print(f'    配置项 model_name = {get_config("model_name")}')
set_config("temperature", 0.3)
print(f'    修改后 temperature = {get_config("temperature")}')
print("    ✅ 模块单例：简单、可靠、Pythonic！")


# ── 实现 ②：__new__ 方法 ──
print("\n  📌 实现 ②：__new__ 方法（经典 OOP）")
print("  " + "-" * 40)


class SingletonViaNew:
    """
    通过重写 __new__ 实现单例

    关键知识点：
      - __new__ 负责「创建」对象（分配内存）
      - __init__ 负责「初始化」对象（设置属性）
      - __new__ 在 __init__ 之前调用
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # super().__new__(cls) 调用 object 的 __new__ 分配内存
            cls._instance = super().__new__(cls) #不能调用自身的__new__方法，否则该函数会自身递归
        return cls._instance

    def __init__(self, value=None):
        # 注意：每次「创建」都会调用 __init__
        # 所以需要用 hasattr 防止重复初始化
        if not hasattr(self, "_initialized"): #检查该类是否含有_initialized属性，若有则不进行初始化
            self.value = value
            self._initialized = True


a = SingletonViaNew("hello")
b = SingletonViaNew("world")  # b 和 a 是同一个对象，value 仍是 "hello"
print(f"    a is b: {a is b}")  # True
print(f"    a.value: {a.value}")  # "hello"（没有因为 b 的初始化改变）


# ── 实现 ③：装饰器 ──
print("\n  📌 实现 ③：装饰器（Pythonic）")
print("  " + "-" * 40)


def singleton(cls):
    """
    将任意类变为单例的装饰器

    原理：
      - instances 字典缓存「类 → 实例」的映射
      - 每次调用被装饰的类时，先查缓存
      - 闭包保存 instances，外部无法访问
    """
    instances = {}

    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs) #这里保证了单例模式，若创建的类已经存在，不能更新参数
        return instances[cls]

    return get_instance


@singleton #这个类使用了singleton包裹 相当于将参数传给get_instance
class DatabaseConnection:
    def __init__(self, host="localhost", port=5432):
        self.host = host
        self.port = port
        self.connected = False

    def connect(self):
        self.connected = True
        return f"已连接 {self.host}:{self.port}"


db1 = DatabaseConnection(host="db-prod", port=5432)
db2 = DatabaseConnection(host="db-test", port=9999)  # 这个参数被忽略！
print(f"    db1 is db2: {db1 is db2}")
print(f"    db1.host: {db1.host}")  # 仍是 db-prod


# ── 实现 ④：元类 ──
print("\n  📌 实现 ④：元类（最强大）")
print("  " + "-" * 40)


class SingletonMeta(type):
    """
    元类版单例

    原理：
      - type 是所有类的「类」，即元类，负责创建类，默认的元类是type，我们在这里重写了元类
      - 当我们写 MyClass() 时，Python 实际调用的是 type.__call__
      - 重写 __call__ 就能拦截实例化过程
    """

    _instances = {} #用于存储所有单例实例，键是类，值是唯一实例

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs) #__call__方法创建类实例
        return cls._instances[cls]


class ModelCache(metaclass=SingletonMeta):
    """模型缓存 —— 用元类确保只加载一次"""

    def __init__(self):
        self.models = {}

    def load(self, model_name):
        if model_name not in self.models:
            # 模拟加载模型（实际项目中这里是加载大模型）
            self.models[model_name] = f"<Model:{model_name}>"
            print(f"      [耗时操作] 正在加载模型 {model_name}...")
        return self.models[model_name]


cache1 = ModelCache()
cache2 = ModelCache()
print(f"    cache1 is cache2: {cache1 is cache2}")
cache1.load("qwen3:0.5b")
cache2.load("qwen3:0.5b")  # 第二次不会重复加载，检测到该类已经有一个实例
print("    ✅ 元类单例：适合框架级别，一次编写到处使用")
print(f"    缓存内容: {cache1.models}")

# ── ✏️ 练习 1：补全单例装饰器 ──
print("\n  ✏️ 练习 1：补全单例装饰器（在下面填空）")
print("  " + "-" * 40)


def singleton_exercise(cls):
    """
    TODO: 补全这个装饰器，使得被装饰的类变成单例

    提示：
      1. 创建一个字典 instances = {}
      2. 定义内部函数 get_instance
      3. 如果 cls 不在 instances 中，创建实例并存入
      4. 返回实例
      5. get_instance 返回 get_instance
    """
    # ===== 你的代码写在这里 =====
    pass  # TODO: 创建 instances 字典

    # ===== 你的代码写在这里 =====
    @wraps(cls)
    def get_instance(*args, **kwargs):
        # ===== 你的代码写在这里 =====
        pass  # TODO: 检查 cls 是否在 instances 中
        # ===== 你的代码写在这里 =====
        pass  # TODO: 返回实例

    return get_instance


# ── 🚀 RAG 实战：Embedding 模型单例缓存 ──
print("\n  🚀 RAG 实战：Embedding 模型单例缓存")
print("  " + "-" * 40)


class EmbeddingModelCache:
    """
    实战场景：
      在你的 RAG 项目中，Embeddings.py 里加载模型（如 text2vec-large-chinese）
      是一个非常耗时的操作。如果每次查询都重新加载模型，系统会非常慢。

    解决方案：用一个单例缓存，确保模型只加载一次。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance

    def get_embedding_model(self, model_name: str):
        """
        获取 Embedding 模型。
        如果是第一次请求这个模型，则「模拟加载」；之后再请求直接返回缓存。
        """
        if model_name not in self._models:
            # 模拟耗时的模型加载过程
            print(f"      ⏳ 正在加载 Embedding 模型: {model_name}...")
            time.sleep(0.3)  # 模拟加载耗时
            self._models[model_name] = {
                "name": model_name,
                "dim": 1024 if "large" in model_name else 768,
                "loaded": True,
            }
            print(f"      ✅ 模型 {model_name} 加载完成！")
        else:
            print(f"      ⚡ 模型 {model_name} 已在缓存中，直接使用！")
        return self._models[model_name]

    def compute_embedding(self, model_name: str, text: str) -> List[float]:
        """模拟计算 embedding（实际项目中调用模型推理）"""
        model = self.get_embedding_model(model_name)
        # 模拟：用文本长度生成假的向量
        import hashlib
        h = hashlib.md5(text.encode()).hexdigest()
        # 取前 dim 个字节转为 0~1 之间的 float
        dim = model["dim"]
        result = []
        for i in range(min(dim, len(h))):
            result.append(int(h[i], 16) / 16.0)
        # 补齐到 dim 长度
        while len(result) < dim:
            result.append(0.0)
        return result


embed_cache = EmbeddingModelCache()
v1 = embed_cache.compute_embedding("text2vec-large-chinese", "保研加分政策")
v2 = embed_cache.compute_embedding("text2vec-large-chinese", "推免细则")
# 第二次调用时，模型已经在缓存中，不会重新加载
print("    ✅ RAG Embedding 模型缓存：大幅提升查询效率！")

# =============================================================================
# 2. 简单工厂模式（Simple Factory）
# =============================================================================

print("\n" + "─" * 60)
print("模式 2：简单工厂模式（Simple Factory）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：客户端代码需要根据参数创建不同类型的对象，但不想关心
      具体的创建逻辑（也不想写一堆 if-else）。

场景：
  • 根据配置选择不同的 LLM（OpenAI / Ollama / 本地模型）
  • 根据文件扩展名选择不同的解析器（PDF / DOCX / TXT）
  • 根据用户等级创建不同的权限对象

核心思想：把「创建对象的逻辑」集中到一个工厂类中。

三种工厂的区别（面试常考）：
  ┌──────────────┬─────────────────────────────────┬──────────────────┐
  │  简单工厂     │  一个工厂类 + 一个创建方法        │  最基础、最常用    │
  │  工厂方法     │  父类定义接口 + 子类决定创建什么   │  对扩展开放        │
  │  抽象工厂     │  创建一系列相关的产品族           │  最复杂、最强大    │
  └──────────────┴─────────────────────────────────┴──────────────────┘
"""

# ── 完整示例 ──
print("\n  📌 完整示例：文档解析器工厂")
print("  " + "-" * 40)


class DocumentParser(ABC):
    """文档解析器抽象基类"""

    @abstractmethod
    def parse(self, file_path: str) -> str:
        pass


class PDFParser(DocumentParser):
    def parse(self, file_path: str) -> str:
        return f"[PDF解析] 从 {file_path} 提取了文本内容"


class DocxParser(DocumentParser):
    def parse(self, file_path: str) -> str:
        return f"[DOCX解析] 从 {file_path} 提取了文本内容"


class TxtParser(DocumentParser):
    def parse(self, file_path: str) -> str:
        return f"[TXT解析] 从 {file_path} 读取了文本内容"


class ParserFactory:
    """简单工厂：根据文件扩展名创建对应的解析器"""

    @staticmethod
    def create_parser(file_path: str) -> DocumentParser:
        ext = file_path.lower().split(".")[-1]
        if ext == "pdf":
            return PDFParser()
        elif ext == "docx":
            return DocxParser()
        elif ext == "txt":
            return TxtParser()
        else:
            raise ValueError(f"不支持的文件类型: .{ext}")


# 客户端代码：不需要知道具体的解析器类名
for f in ["论文.pdf", "报告.docx", "笔记.txt"]:
    parser = ParserFactory.create_parser(f)
    print(f"    解析 {f}: {parser.parse(f)}")


# ── ✏️ 练习 2：补全 LLM 客户端工厂 ──
print("\n  ✏️ 练习 2：补全 LLM 客户端工厂")
print("  " + "-" * 40)


class LLMClient(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str) -> str:
        return f"[OpenAI:{self.model}] 回复: 根据{prompt[:20]}...生成的回答"


class OllamaClient(LLMClient):
    def __init__(self, model: str = "qwen3:0.5b"):
        self.model = model

    def chat(self, prompt: str) -> str:
        return f"[Ollama:{self.model}] 回复: 根据{prompt[:20]}...生成的回答"


class LLMFactory:
    """LLM 客户端工厂"""

    @staticmethod
    def create(config: dict) -> LLMClient:
        """
        根据配置字典创建对应的 LLM 客户端

        配置格式：
          {"provider": "openai", "api_key": "sk-xxx", "model": "gpt-4"}
          {"provider": "ollama", "model": "qwen3:0.5b"}

        TODO: 补全下面的创建逻辑
        """
        provider = config.get("provider", "").lower()

        # ===== 你的代码写在这里 =====
        pass  # TODO: 根据 provider 创建对应的 LLM 客户端
        # ===== 你的代码写在这里 =====
        pass  # TODO: 处理其他 provider
        

# ── 🚀 RAG 实战：可配置的多 LLM 切换 ──
print("\n  🚀 RAG 实战：根据 .env 配置切换 LLM")
print("  " + "-" * 40)

"""
实战场景：
  你的 LLM.py 中目前可能只接入了一个模型。使用工厂模式后，
  可以通过修改 .env 文件中的配置，轻松在 OpenAI / Ollama / 其他模型之间切换，
  而不需要修改业务代码。

.env 示例：
  LLM_PROVIDER=ollama
  LLM_MODEL=qwen3:0.5b
  LLM_API_KEY=sk-xxx
"""


class RAGLLMService:
    """
    RAG 项目的 LLM 服务类
    使用工厂模式解耦「模型选择」和「业务逻辑」
    """

    def __init__(self, provider: str = "ollama", model: str = "qwen3:0.5b",
                 api_key: str = ""):
        config = {
            "provider": provider,
            "model": model,
            "api_key": api_key,
        }
        self.client = LLMFactory.create(config)

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""用户问题：{question}
参考材料：{context}
请基于以上材料回答问题。"""
        return self.client.chat(prompt)


# 模拟使用
service = RAGLLMService(provider="ollama", model="qwen3:0.5b")
answer = service.generate_answer("加分政策是什么", "保研加分包括竞赛加分...")
print(f"    RAG 回答: {answer}")


# =============================================================================
# 3. 工厂方法模式（Factory Method Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 3：工厂方法模式（Factory Method Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

与简单工厂的区别：
  简单工厂：一个工厂类根据参数创建不同产品（违反开闭原则，加新产品要改工厂）
  工厂方法：定义抽象的工厂接口，让子类决定实例化哪个具体产品（对扩展开放）

开闭原则（OCP）：对扩展开放，对修改关闭。
  → 加新产品时，只需新增子类，不需要修改已有代码。

场景：
  • 不同检索策略：向量检索 / BM25 关键词检索 / 混合检索
  • 不同存储后端：本地文件 / MySQL / Elasticsearch
  • 不同评测指标：MRR / NDCG / Recall@K
"""

# ── 完整示例 ──
print("\n  📌 完整示例：物流运输")
print("  " + "-" * 40)


class Transport(ABC):
    @abstractmethod
    def deliver(self) -> str:
        pass


class Truck(Transport):
    def deliver(self):
        return "🚛 卡车陆运配送"


class Ship(Transport):
    def deliver(self):
        return "🚢 货轮海运配送"


class Airplane(Transport):
    def deliver(self):
        return "✈️ 飞机空运配送"


class Logistics(ABC):
    """
    物流基类 —— 定义了「工厂方法」接口
    子类需要实现 create_transport()
    """

    @abstractmethod
    def create_transport(self) -> Transport:
        """工厂方法：由子类决定创建哪种运输工具"""
        pass

    def plan_delivery(self) -> str:
        """模板方法：固定的配送流程"""
        transport = self.create_transport()
        return f"物流计划：使用 {transport.deliver()}"


class RoadLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Truck()


class SeaLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Ship()


class AirLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Airplane()


# 测试
for logistics in [RoadLogistics(), SeaLogistics(), AirLogistics()]:
    print(f"    {logistics.plan_delivery()}")


# ── ✏️ 练习 3：补全检索策略工厂 ──
print("\n  ✏️ 练习 3：补全检索策略（工厂方法）")
print("  " + "-" * 40)


class SearchStrategy(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[dict]:
        """返回检索结果列表"""
        pass


class VectorSearch(SearchStrategy):
    def search(self, query: str, top_k: int) -> List[dict]:
        return [
            {"id": i, "method": "向量检索", "score": 0.9 - i * 0.1,
             "content": f"向量结果{i}"}
            for i in range(top_k)
        ]


class KeywordSearch(SearchStrategy):
    def search(self, query: str, top_k: int) -> List[dict]:
        return [
            {"id": i, "method": "关键词检索", "score": 0.8 - i * 0.15,
             "content": f"关键词结果{i}"}
            for i in range(top_k)
        ]


class HybridSearch(SearchStrategy):
    """混合检索：结合向量检索和关键词检索的结果"""

    def __init__(self):
        self.vector = VectorSearch()
        self.keyword = KeywordSearch()

    def search(self, query: str, top_k: int) -> List[dict]:
        vec_results = self.vector.search(query, top_k)
        kw_results = self.keyword.search(query, top_k)
        # 简单合并（实际项目中会用 RRF 等算法融合）
        combined = vec_results + kw_results
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]


class SearchFactory(ABC):
    """检索工厂抽象类"""

    @abstractmethod
    def create_search(self) -> SearchStrategy:
        """
        TODO: 子类需要实现这个方法，返回具体的检索策略
        """
        pass

    def execute_search(self, query: str, top_k: int = 3) -> List[dict]:
        """执行检索（不需要子类重写）"""
        strategy = self.create_search()
        return strategy.search(query, top_k)


# ===== 你的代码写在这里：实现具体的检索工厂 =====

class VectorSearchFactory(SearchFactory):
    """向量检索工厂"""
    # TODO: 实现 create_search 方法
    def create_search(self) -> SearchStrategy:
        # ===== 你的代码 =====
        pass  # TODO: 返回 VectorSearch 实例


class KeywordSearchFactory(SearchFactory):
    """关键词检索工厂"""
    # TODO: 实现 create_search 方法
    def create_search(self) -> SearchStrategy:
        # ===== 你的代码 =====
        pass  # TODO: 返回 KeywordSearch 实例


class HybridSearchFactory(SearchFactory):
    """混合检索工厂"""
    # TODO: 实现 create_search 方法
    def create_search(self) -> SearchStrategy:
        # ===== 你的代码 =====
        pass  # TODO: 返回 HybridSearch 实例


# ── 🚀 RAG 实战：可切换的检索策略 ──
print("\n  🚀 RAG 实战：根据场景选择检索策略")
print("  " + "-" * 40)

"""
实战场景：
  你的 HybridSearch.py 目前混合了多种检索方式。使用工厂方法模式后：
    - 精确查询（如查规定条文） → 关键词检索
    - 语义查询（如查相似内容） → 向量检索
    - 综合查询                   → 混合检索

  业务代码只需选择对应的工厂，不关心底层实现。
"""


class RAGRetrievalService:
    """RAG 检索服务 —— 使用工厂方法切换策略"""

    def __init__(self, factory: SearchFactory):
        self.factory = factory

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        return self.factory.execute_search(query, top_k)


# 根据查询类型选择策略
query_type = "hybrid"  # 可以是 "vector" / "keyword" / "hybrid"
if query_type == "vector":
    factory = VectorSearchFactory()
elif query_type == "keyword":
    factory = KeywordSearchFactory()
else:
    factory = HybridSearchFactory()

service = RAGRetrievalService(factory)
results = service.retrieve("保研加分政策", top_k=3)
print(f"    查询 '保研加分政策'（{query_type}模式）:")
for r in results:
    print(f"      [{r['method']}] score={r['score']:.2f} → {r['content']}")


# =============================================================================
# 4. 抽象工厂模式（Abstract Factory Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 4：抽象工厂模式（Abstract Factory Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

与工厂方法的区别：
  工厂方法：创建一个产品
  抽象工厂：创建「一族」相关的产品

场景：
  • 跨平台 UI：一个工厂创建整套 UI 组件（按钮+文本框+菜单）
  • RAG Pipeline：一个工厂创建一整套组件（Embedding 模型 + LLM + Reranker）
  • 数据库层：一个工厂创建整套 DAO（UserDAO + OrderDAO + ProductDAO）

核心思想：确保同一工厂创建的产品之间是「兼容」的。
  比如 Mac 工厂不会创建出 Windows 风格的按钮。
"""

# ── 完整示例：跨平台 UI ──
print("\n  📌 完整示例：跨平台 UI 组件")
print("  " + "-" * 40)


class Button(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class TextBox(ABC):
    @abstractmethod
    def render(self) -> str:
        pass


class MacButton(Button):
    def render(self):
        return "🍎 Mac 风格按钮 [圆角、灰色]"


class MacTextBox(TextBox):
    def render(self):
        return "🍎 Mac 风格文本框 [半透明、阴影]"


class WinButton(Button):
    def render(self):
        return "🪟 Windows 风格按钮 [方角、蓝色]"


class WinTextBox(TextBox):
    def render(self):
        return "🪟 Windows 风格文本框 [白色、边框]"


class GUIFactory(ABC):
    """抽象工厂：定义创建产品族的方法"""

    @abstractmethod
    def create_button(self) -> Button:
        pass

    @abstractmethod
    def create_textbox(self) -> TextBox:
        pass


class MacFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacButton()

    def create_textbox(self) -> TextBox:
        return MacTextBox()


class WinFactory(GUIFactory):
    def create_button(self) -> Button:
        return WinButton()

    def create_textbox(self) -> TextBox:
        return WinTextBox()


def build_ui(factory: GUIFactory):
    """客户端代码：不关心具体平台，只使用抽象接口"""
    button = factory.create_button()
    textbox = factory.create_textbox()
    return f"UI 构建完成:\n      {button.render()}\n      {textbox.render()}"


print(f"    {build_ui(MacFactory())}")
print(f"    {build_ui(WinFactory())}")


# ── 🚀 RAG 实战：RAG Pipeline 组件族 ──
print("\n  🚀 RAG 实战：RAG Pipeline 整套组件抽象工厂")
print("  " + "-" * 40)

"""
实战场景：
  你的 RAG 项目可能需要多种配置：
    方案A（轻量级）：本地 Embedding + Ollama LLM + 无 Reranker
    方案B（高精度）：  OpenAI Embedding + GPT-4 + Cohere Reranker
  
  使用抽象工厂，切换方案只需换一个工厂实例。
"""


class EmbeddingService(ABC):
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        pass


class RerankerService(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        pass


class LocalEmbedding(EmbeddingService):
    def encode(self, text: str) -> List[float]:
        return [0.1 * (ord(c) % 10) for c in text[:10]]


class OpenAIEmbedding(EmbeddingService):
    def encode(self, text: str) -> List[float]:
        return [0.2 * (ord(c) % 10) for c in text[:10]]


class NoReranker(RerankerService):
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        return [1.0 / (i + 1) for i in range(len(documents))]


class CohereReranker(RerankerService):
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        # 模拟高精度重排序
        return [0.95 - i * 0.15 for i in range(len(documents))]


class RAGPipelineFactory(ABC):
    """RAG Pipeline 抽象工厂"""

    @abstractmethod
    def create_embedding(self) -> EmbeddingService:
        pass

    @abstractmethod
    def create_llm(self) -> LLMClient:
        pass

    @abstractmethod
    def create_reranker(self) -> RerankerService:
        pass


class LitePipelineFactory(RAGPipelineFactory):
    """轻量级 Pipeline"""

    def create_embedding(self) -> EmbeddingService:
        return LocalEmbedding()

    def create_llm(self) -> LLMClient:
        return OllamaClient(model="qwen3:0.5b")

    def create_reranker(self) -> RerankerService:
        return NoReranker()


class ProPipelineFactory(RAGPipelineFactory):
    """高精度 Pipeline"""

    def create_embedding(self) -> EmbeddingService:
        return OpenAIEmbedding()

    def create_llm(self) -> LLMClient:
        return OpenAIClient(api_key="sk-xxx", model="gpt-4")

    def create_reranker(self) -> RerankerService:
        return CohereReranker()


def run_rag_pipeline(factory: RAGPipelineFactory, query: str,
                     docs: List[str]) -> dict:
    """运行完整的 RAG Pipeline"""
    embedding = factory.create_embedding()
    llm = factory.create_llm()
    reranker = factory.create_reranker()

    query_vec = embedding.encode(query)
    scores = reranker.rerank(query, docs)
    context = "\n".join(docs)
    answer = llm.chat(f"问题：{query}\n上下文：{context}")

    return {
        "embedding_dim": len(query_vec),
        "rerank_scores": scores,
        "answer": answer,
    }


lite_result = run_rag_pipeline(
    LitePipelineFactory(),
    "保研加分", ["文档1内容", "文档2内容", "文档3内容"]
)
print(f"    轻量级 Pipeline 结果: embedding_dim={lite_result['embedding_dim']}, "
      f"scores={[f'{s:.2f}' for s in lite_result['rerank_scores']]}")


# =============================================================================
# 5. 建造者模式（Builder Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 5：建造者模式（Builder Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：一个对象的构造过程很复杂（很多可选参数、需要分步构建），
      直接写在构造函数里会导致参数列表过长、代码难以阅读。

场景：
  • 构建复杂的 Prompt（系统指令 + 上下文 + Few-shot 示例 + 输出格式）
  • 构建复杂的 HTTP 请求（URL + Headers + Body + Timeout + Retry）
  • 构建复杂的查询条件（多个 filter + sort + pagination）

核心思想：把「构建过程」和「最终对象」分离。
  通过链式调用逐步设置参数，最后调用 build() 生成对象。

与工厂的区别：
  工厂：一步创建（create() → 对象）
  建造者：分步构建（.setA().setB().build() → 对象）
"""

# ── 完整示例 ──
print("\n  📌 完整示例：HTTP 请求建造者")
print("  " + "-" * 40)


class HTTPRequest:
    def __init__(self):
        self.url = ""
        self.method = "GET"
        self.headers = {}
        self.body = None
        self.timeout = 30

    def __repr__(self):
        return (f"HTTPRequest(method={self.method}, url={self.url}, "
                f"timeout={self.timeout}, headers={list(self.headers.keys())})")


class RequestBuilder:
    def __init__(self):
        self.request = HTTPRequest()

    def set_url(self, url: str):
        self.request.url = url
        return self  # 返回 self 实现链式调用

    def set_method(self, method: str):
        self.request.method = method.upper()
        return self

    def add_header(self, key: str, value: str):
        self.request.headers[key] = value
        return self

    def set_body(self, body: dict):
        self.request.body = body
        return self

    def set_timeout(self, timeout: int):
        self.request.timeout = timeout
        return self

    def build(self) -> HTTPRequest:
        if not self.request.url:
            raise ValueError("URL 不能为空")
        return self.request


# 链式调用
req = (RequestBuilder()
       .set_url("https://api.openai.com/v1/chat/completions")
       .set_method("POST")
       .add_header("Authorization", "Bearer sk-xxx")
       .add_header("Content-Type", "application/json")
       .set_body({"model": "gpt-4", "messages": []})
       .set_timeout(60)
       .build())
print(f"    {req}")


# ── ✏️ 练习 4：补全 Prompt 建造者 ──
print("\n  ✏️ 练习 4：补全 Prompt 建造者")
print("  " + "-" * 40)


class Prompt:
    """RAG Prompt 对象"""

    def __init__(self):
        self.system_prompt = ""
        self.instructions = []
        self.contexts = []
        self.examples = []
        self.question = ""
        self.output_format = ""

    def to_string(self) -> str:
        """将 Prompt 组装为最终字符串"""
        parts = []
        if self.system_prompt:
            parts.append(f"【系统指令】\n{self.system_prompt}")
        if self.instructions:
            parts.append(f"【具体要求】\n" + "\n".join(
                f"  {i + 1}. {inst}" for i, inst in enumerate(self.instructions)))
        if self.examples:
            parts.append(f"【示例】\n" + "\n---\n".join(self.examples))
        if self.contexts:
            parts.append(f"【参考材料】\n" + "\n---\n".join(self.contexts))
        if self.question:
            parts.append(f"【用户问题】\n{self.question}")
        if self.output_format:
            parts.append(f"【输出格式】\n{self.output_format}")
        return "\n\n".join(parts)


class PromptBuilder:
    """
    Prompt 建造者 — 分步构建复杂的 RAG Prompt

    TODO: 补全下面每个方法的返回值和 build 方法
    """

    def __init__(self):
        self.prompt = Prompt()

    def set_system_prompt(self, text: str):
        self.prompt.system_prompt = text
        # ===== 你的代码：补全返回值以支持链式调用 =====
        pass  # TODO: 返回 self

    def add_instruction(self, instruction: str):
        self.prompt.instructions.append(instruction)
        # ===== 你的代码 =====
        pass  # TODO: 返回 self

    def add_context(self, context: str):
        self.prompt.contexts.append(context)
        # ===== 你的代码 =====
        pass  # TODO: 返回 self

    def add_example(self, example_qa: str):
        self.prompt.examples.append(example_qa)
        # ===== 你的代码 =====
        pass  # TODO: 返回 self

    def set_question(self, question: str):
        self.prompt.question = question
        # ===== 你的代码 =====
        pass  # TODO: 返回 self

    def set_output_format(self, fmt: str):
        self.prompt.output_format = fmt
        # ===== 你的代码 =====
        pass  # TODO: 返回 self

    def build(self) -> Prompt:
        # ===== 你的代码：验证必要字段 =====
        pass  # TODO: 验证 question 不为空，返回 self.prompt


# ── 🚀 RAG 实战：用于评测的复杂 Prompt 组装 ──
print("\n  🚀 RAG 实战：构建评测用的 RAG Prompt")
print("  " + "-" * 40)

"""
实战场景：
  你的 run_eval.py 在评测时需要组装复杂的 Prompt：
    - 系统指令（角色设定）
    - 具体答题要求（如"只基于材料回答"）
    - Few-shot 示例（帮助模型理解任务格式）
    - 多条 chunk 上下文
    - 用户问题
    - 输出格式要求（JSON 格式）
"""

eval_prompt = (PromptBuilder()
               .set_system_prompt(
    "你是一个专业的文档问答助手。请严格基于提供的参考材料回答问题。"
)
               .add_instruction("只使用参考材料中的信息，不要编造")
               .add_instruction("如果材料中没有相关信息，明确回答「材料中未提及」")
               .add_instruction("回答要简洁准确，不超过 200 字")
               .add_example(
    "问题：保研加分包括哪些竞赛？\n"
    "回答：根据材料，保研加分包括全国大学生电子设计竞赛、"
    "ACM-ICPC 国际大学生程序设计竞赛等国家级竞赛。"
)
               .add_context("chunk 0 (score:0.95): 电院关于高水平科技竞赛获奖...")
               .add_context("chunk 1 (score:0.87): 中南财保研工作管理办法...")
               .set_question("加分政策具体是什么？")
               .set_output_format(
    '请以 JSON 格式输出：{"answer": "...", "sources": [...]}'
)
               .build())

print("    构建的评测 Prompt:")
print("    " + "-" * 50)
for line in eval_prompt.to_string().split("\n")[:8]:
    print(f"    {line}")
print("    ...")
print("    ✅ Prompt 建造者：清晰、可维护、可复用！")

# =============================================================================
# 🧱 第二部分：结构型模式
# =============================================================================

print("\n" + "=" * 60)
print("🧱 第二部分：结构型模式")
print("=" * 60)

# =============================================================================
# 6. 装饰器模式（Decorator Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 6：装饰器模式（Decorator Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Python 中最重要、最常用的模式之一！

问题：如何在不修改原函数/类的前提下，为其添加额外的功能？

场景：
  • 为函数添加计时功能
  • 为函数添加日志记录
  • 为 API 调用添加重试机制
  • 为函数添加缓存
  • 权限验证（如 Flask/Django 的 @login_required）

Python 的 @ 语法糖让装饰器用起来非常自然。
这也是 Python 区别于 Java/C++ 的一大特色。

核心知识点：
  1. 函数是一等公民（可以作为参数传递）
  2. 闭包（内部函数记住外部作用域的变量）
  3. @wraps 保留原函数的元信息（__name__, __doc__ 等）
"""

# ── 基础：函数装饰器 ──
print("\n  📌 基础：自己写一个函数装饰器")
print("  " + "-" * 40)


def timer(func):
    """计时装饰器：打印函数的执行时间"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"      ⏱️ {func.__name__} 耗时: {elapsed:.4f} 秒")
        return result

    return wrapper


@timer
def slow_computation(n: int) -> int:
    """模拟耗时计算"""
    time.sleep(0.3)
    return n * n


result = slow_computation(10)
print(f"    计算结果: {result}")


# ── 带参数的装饰器 ──
print("\n  📌 进阶：带参数的装饰器（三层嵌套）")
print("  " + "-" * 40)


def retry(max_attempts: int = 3, delay: float = 0.5):
    """
    重试装饰器：函数调用失败时自动重试

    三层嵌套原理：
      retry(max_attempts, delay) → 返回 decorator
      decorator(func)            → 返回 wrapper
      wrapper(*args, **kwargs)   → 执行实际逻辑
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        print(f"      ⚠️ 第 {attempt} 次失败: {e}，"
                              f"{delay}秒后重试...")
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


fail_count = 0


@retry(max_attempts=3, delay=0.2)
def unstable_api_call():
    """模拟不稳定的 API 调用（前两次失败，第三次成功）"""
    global fail_count
    fail_count += 1
    if fail_count < 3:
        raise ConnectionError(f"网络错误 (第{fail_count}次)")
    return "API 调用成功！"


print(f"    不稳定 API 结果: {unstable_api_call()}")


# ── ✏️ 练习 5：写一个缓存装饰器 ──
print("\n  ✏️ 练习 5：补全缓存装饰器")
print("  " + "-" * 40)


def cache(func):
    """
    缓存装饰器：将函数结果缓存起来

    要求：
      - 用字典保存 (args, kwargs) → 结果的映射
      - 如果参数相同，直接返回缓存结果
      - 使用 @wraps 保留原函数信息

    TODO: 补全这个装饰器
    """
    cache_dict = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # ===== 你的代码：构造缓存键 =====
        # 提示：kwargs 需要先排序再转成元组，保证顺序无关
        pass  # TODO: 构造缓存键 key
        # ===== 你的代码：检查缓存 =====
        pass  # TODO: 如果 key 在 cache_dict 中，返回缓存结果
        # ===== 你的代码：计算并缓存 =====
        pass  # TODO: 调用原函数，缓存结果，返回

    return wrapper


@cache
def expensive_embedding(text: str, model: str = "default") -> int:
    """模拟昂贵的 embedding 计算"""
    time.sleep(0.2)
    return len(text) * 10


print(f"    第一次: {expensive_embedding('hello')}")
print(f"    第二次(应命中缓存): {expensive_embedding('hello')}")
print(f"    不同参数: {expensive_embedding('world')}")


# ── 类装饰器 ──
print("\n  📌 类装饰器：给类添加功能")
print("  " + "-" * 40)


def add_repr(cls):
    """为类自动添加 __repr__ 方法"""

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"

    cls.__repr__ = __repr__
    return cls


@add_repr
class RetrievalResult:
    def __init__(self, query: str, docs: list, scores: list):
        self.query = query
        self.docs = docs
        self.scores = scores


r = RetrievalResult("测试", ["doc1", "doc2"], [0.9, 0.8])
print(f"    自动生成的 repr: {r}")


# ── 🚀 RAG 实战：为检索函数添加日志和计时 ──
print("\n  🚀 RAG 实战：用装饰器增强检索和 LLM 调用")
print("  " + "-" * 40)

"""
实战场景：
  在 eval_logger.py 中，你可能需要给每次检索/LLM 调用添加日志。
  使用装饰器可以无侵入地实现：
    @log_call
    @timer
    def search(query): ...
"""


def log_call(func):
    """记录函数调用日志"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        timestamp = time.strftime("%H:%M:%S")
        short_args = str(args)[:60] if args else ""
        short_kwargs = str(kwargs)[:60] if kwargs else ""
        print(f"      📝 [{timestamp}] 调用 {func.__name__}"
              f"({short_args}{short_kwargs})")
        result = func(*args, **kwargs)
        return result

    return wrapper


@log_call
@timer
@cache  # 三层装饰器：日志 → 计时 → 缓存
def retrieve_documents(query: str, top_k: int = 5) -> List[str]:
    """模拟 RAG 检索"""
    time.sleep(0.15)
    return [f"文档{i}: 与'{query}'相关的内容" for i in range(top_k)]


docs1 = retrieve_documents("保研加分", top_k=3)
docs2 = retrieve_documents("保研加分", top_k=3)  # 缓存命中
print(f"    检索结果数: {len(docs1)}")


# =============================================================================
# 7. 适配器模式（Adapter Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 7：适配器模式（Adapter Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：如何让两个不兼容的接口协同工作？

就像出国旅行时的「电源转换插头」：
  中国的插头（国标） → 适配器 → 欧洲的插座（欧标）

场景：
  • 统一不同 LLM API 的调用方式（OpenAI / Claude / Ollama 接口不同）
  • 统一不同数据库的查询接口
  • 统一不同文件格式的读取接口

核心思想：不修改原有代码，而是加一个中间层做「接口转换」。
"""

# ── 完整示例 ──
print("\n  📌 完整示例：统一不同 LLM 的调用接口")
print("  " + "-" * 40)


# 模拟第三方库（接口不同）
class OpenAIAPI:
    """OpenAI SDK 风格"""
    def create_chat_completion(self, messages: list, model: str,
                               temperature: float) -> dict:
        return {
            "choices": [{"message": {"content":
                                     f"[GPT-{model}] 回答: ..."}}]
        }


class ClaudeAPI:
    """Anthropic SDK 风格（接口不同！）"""
    def generate(self, prompt: str, model: str,
                 max_tokens: int, temp: float) -> dict:
        return {
            "completion": f"[Claude-{model}] 回答: ..."
        }


# 统一的接口
class UnifiedLLM(ABC):
    @abstractmethod
    def chat(self, prompt: str) -> str:
        pass


class OpenAIAdapter(UnifiedLLM):
    """将 OpenAI API 适配到统一接口"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api = OpenAIAPI()
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str) -> str:
        response = self.api.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"]


class ClaudeAdapter(UnifiedLLM):
    """将 Claude API 适配到统一接口"""

    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        self.api = ClaudeAPI()
        self.api_key = api_key
        self.model = model

    def chat(self, prompt: str) -> str:
        response = self.api.generate(
            prompt=prompt,
            model=self.model,
            max_tokens=2048,
            temp=0.7,
        )
        return response["completion"]


# 客户端代码：不再关心底层是哪个 API
def use_llm(llm: UnifiedLLM, question: str) -> str:
    return llm.chat(question)


print(f"    {use_llm(OpenAIAdapter('sk-xxx'), '你好')}")
print(f"    {use_llm(ClaudeAdapter('sk-xxx'), '你好')}")


# ── ✏️ 练习 6：补全 Ollama 适配器 ──
print("\n  ✏️ 练习 6：补全 Ollama 适配器")
print("  " + "-" * 40)


# 模拟 Ollama API（和前面两种都不一样）
class OllamaAPI:
    def ask(self, question: str, system: str = "",
            model_name: str = "llama3") -> str:
        return f"[Ollama-{model_name}] 对 '{question[:15]}...' 的回答"


class OllamaAdapter(UnifiedLLM):
    """
    TODO: 将 OllamaAPI 适配到 UnifiedLLM 接口

    提示：
      1. 在 __init__ 中创建 OllamaAPI 实例
      2. 实现 chat 方法，调用 self.api.ask()
    """

    def __init__(self, model: str = "qwen3:0.5b"):
        # ===== 你的代码 =====
        pass  # TODO: 创建 OllamaAPI 实例，保存 model

    def chat(self, prompt: str) -> str:
        # ===== 你的代码 =====
        pass  # TODO: 调用 self.api.ask() 并返回结果


print(f"    {use_llm(OllamaAdapter('qwen3:0.5b'), '保研加分政策是什么')}")

# ── 🚀 RAG 实战：统一的 LLM 调用层 ──
print("\n  🚀 RAG 实战：不管用哪个模型，业务代码不变")
print("  " + "-" * 40)

"""
实战场景：
  你的 LLM.py 中，如果要在 OpenAI 和 Ollama 之间切换，
  使用适配器模式后，只需改配置，业务代码不用动。
  
  更进一步的实战价值：
    当你需要在评测中对比不同模型时，只需传入不同的适配器即可。
"""


class RAGApplication:
    """RAG 应用 —— 完全解耦 LLM 实现"""

    def __init__(self, llm: UnifiedLLM):
        self.llm = llm

    def answer(self, question: str, retrieved_docs: List[str]) -> str:
        context = "\n".join(retrieved_docs)
        prompt = f"问题：{question}\n材料：{context}"
        return self.llm.chat(prompt)


# 评测时切换模型只需改这一行
app = RAGApplication(OllamaAdapter("qwen3:0.5b"))
ans = app.answer("加分政策", ["文档1: 保研加分...", "文档2: 竞赛..."])


# =============================================================================
# 8. 代理模式（Proxy Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 8：代理模式（Proxy Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：如何在不修改原始对象的前提下，控制对它的访问？

就像一个明星的经纪人：
  你想找明星合作 → 经纪人（代理）先过滤 → 决定是否转达

代理的常见类型：
  • 远程代理 — 隐藏网络通信细节
  • 虚拟代理 — 延迟加载大对象（懒加载）
  • 保护代理 — 权限控制
  • 缓存代理 — 缓存结果
  • 限流代理 — 控制调用频率

与装饰器的区别：
  装饰器：增强功能（加日志、计时等）
  代理：控制访问（可能完全阻止调用）
"""

# ── 完整示例：限流代理 ──
print("\n  📌 完整示例：API 调用限流代理")
print("  " + "-" * 40)


class RateLimiter:
    """简单的令牌桶限流器"""

    def __init__(self, max_calls_per_second: float):
        self.interval = 1.0 / max_calls_per_second
        self.last_call_time = 0.0

    def acquire(self) -> bool:
        now = time.time()
        if now - self.last_call_time >= self.interval:
            self.last_call_time = now
            return True
        return False


class LLMServiceProxy(UnifiedLLM):
    """LLM 服务的限流代理"""

    def __init__(self, real_llm: UnifiedLLM,
                 max_calls_per_second: float = 2.0):
        self._real_llm = real_llm
        self._rate_limiter = RateLimiter(max_calls_per_second)

    def chat(self, prompt: str) -> str:
        if not self._rate_limiter.acquire():
            raise RuntimeError(
                "请求过于频繁，请稍后再试（限流保护）")
        print(f"      ✅ 限流代理放行: 转发到 {type(self._real_llm).__name__}")
        return self._real_llm.chat(prompt)


real_llm = OllamaAdapter("qwen3:0.5b")
proxy = LLMServiceProxy(real_llm, max_calls_per_second=10)
print(f"    第一问: {proxy.chat('问题1')}")
time.sleep(0.15)  # 等待一小段时间，避免触发限流
print(f"    第二问: {proxy.chat('问题2')}")


# ── 🚀 RAG 实战：LLM 调用代理（日志 + 限流 + 缓存） ──
print("\n  🚀 RAG 实战：增强版 LLM 代理")
print("  " + "-" * 40)


class EnhancedLLMProxy(UnifiedLLM):
    """增强版代理：组合了缓存、日志、限流功能"""

    def __init__(self, real_llm: UnifiedLLM,
                 max_calls_per_second: float = 3.0):
        self._real_llm = real_llm
        self._cache = {}
        self._limiter = RateLimiter(max_calls_per_second)
        self._call_count = 0

    def chat(self, prompt: str) -> str:
        # 1. 查缓存
        if prompt in self._cache:
            print(f"      ⚡ 代理缓存命中")
            return self._cache[prompt]

        # 2. 限流
        if not self._limiter.acquire():
            time.sleep(0.1)  # 等一等再试

        # 3. 转发前记录日志
        self._call_count += 1
        print(f"      📝 [第{self._call_count}次调用] "
              f"prompt 长度: {len(prompt)}")

        # 4. 调用真实服务
        result = self._real_llm.chat(prompt)

        # 5. 存缓存
        self._cache[prompt] = result
        return result


enhanced_llm = EnhancedLLMProxy(OllamaAdapter("qwen3:0.5b"))
print(f"    测试 1: {enhanced_llm.chat('你好')}")
print(f"    测试 2(缓存): {enhanced_llm.chat('你好')}")


# =============================================================================
# 9. 外观模式（Facade Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 9：外观模式（Facade Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：一个子系统非常复杂，有多个类需要协作。如何简化客户端的使用？

就像一个「一键启动」按钮：
  你不关心引擎、油路、电路的具体细节，
  只需按下启动键，汽车自动完成所有步骤。

场景：
  • RAG 系统：一次调用完成「检索 → 重排 → 生成」
  • 电商下单：一次调用完成「验证库存 → 计算价格 → 创建订单 → 扣款」
  • 编译系统：一次调用完成「预处理 → 编译 → 汇编 → 链接」

核心思想：为复杂子系统提供一个简单的统一接口。
"""

# ── 🚀 RAG 实战：RAG 门面 ──
print("\n  🚀 RAG 实战：一行代码完成检索+生成")
print("  " + "-" * 40)

"""
实战场景：
  你的 RAG 项目目前需要手动调用多个步骤：
    VectorBase.search() → Reranker.rerank() → LLM.generate()
  
  使用外观模式封装后，外部只需一行代码。
"""


class RAGFacade:
    """RAG 门面：封装检索 + 重排 + 生成的全流程"""

    def __init__(self,
                 search_factory: SearchFactory,
                 llm: UnifiedLLM,
                 reranker: Optional[RerankerService] = None):
        self.searcher = search_factory.create_search()
        self.llm = llm
        self.reranker = reranker

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        一键问答：传入问题，返回答案

        内部流程：
          1. 检索 top_k 个相关文档
          2. （可选）重排序
          3. 调用 LLM 生成答案
        """
        # 1. 检索
        docs = self.searcher.search(question, top_k)

        # 2. 重排（如果有 reranker）
        if self.reranker:
            contents = [d["content"] for d in docs]
            scores = self.reranker.rerank(question, contents)
            # 按新分数排序
            for d, s in zip(docs, scores):
                d["score"] = s
            docs.sort(key=lambda x: x["score"], reverse=True)

        # 3. 生成
        context = "\n".join(d["content"] for d in docs)
        answer = self.llm.chat(
            f"问题：{question}\n参考材料：{context}"
        )

        return {
            "question": question,
            "retrieved_docs": docs,
            "answer": answer,
        }


# 使用门面
facade = RAGFacade(
    search_factory=HybridSearchFactory(),
    llm=OllamaAdapter("qwen3:0.5b"),
    reranker=CohereReranker(),
)
result = facade.ask("保研加分政策", top_k=3)
print(f"    问题: {result['question']}")
print(f"    检索到: {len(result['retrieved_docs'])} 篇文档")
print(f"    回答: {result['answer'][:100]}...")
print("    ✅ RAG 门面：一行 ask()，完成全部流程！")


# =============================================================================
# 🧠 第三部分：行为型模式
# =============================================================================

print("\n" + "=" * 60)
print("🧠 第三部分：行为型模式")
print("=" * 60)

# =============================================================================
# 10. 策略模式（Strategy Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 10：策略模式（Strategy Pattern）⭐ 最实用")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：一个功能有多种实现方式，如何让它们可以灵活切换？

就像手机输入法：
  你可以随时在拼音 / 手写 / 语音输入之间切换，
  但打字的「结果（文字）」是一样的。

场景：
  • 排序算法可切换（快速排序 / 归并排序 / 堆排序）
  • 支付方式可切换（微信 / 支付宝 / 银行卡）
  • 相似度计算可切换（余弦 / 欧氏 / 曼哈顿距离）
  • Reranker 可切换（无 / Cross-Encoder / LLM-based）

核心思想：定义一组算法，把它们封装成可互换的策略类。

与工厂模式的区别：
  工厂：关注「创建哪个对象」
  策略：关注「使用哪个算法」
"""

# ── 完整示例 ──
print("\n  📌 完整示例：排序策略")
print("  " + "-" * 40)


class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass


class QuickSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("      使用快速排序")
        return sorted(data)


class BubbleSort(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("      使用冒泡排序")
        result = list(data)
        n = len(result)
        for i in range(n):
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result


class DataSorter:
    """上下文类：使用策略"""

    def __init__(self, strategy: SortStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        """运行时切换策略"""
        self.strategy = strategy

    def execute_sort(self, data: List[int]) -> List[int]:
        return self.strategy.sort(data)


sorter = DataSorter(QuickSort())
print(f"    快排: {sorter.execute_sort([3, 1, 4, 1, 5])}")
sorter.set_strategy(BubbleSort())
print(f"    冒泡: {sorter.execute_sort([3, 1, 4, 1, 5])}")


# ── ✏️ 练习 7：补全相似度策略 ──
print("\n  ✏️ 练习 7：补全相似度计算策略")
print("  " + "-" * 40)


class SimilarityStrategy(ABC):
    @abstractmethod
    def compute(self, v1: List[float], v2: List[float]) -> float:
        pass


class CosineSimilarity(SimilarityStrategy):
    def compute(self, v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)


class EuclideanDistance(SimilarityStrategy):
    """
    欧氏距离（越小越相似）
    公式：sqrt( sum( (a-b)^2 ) )

    TODO: 补全 compute 方法
    """
    # ===== 你的代码 =====
    pass  # TODO: 实现欧氏距离，转换为相似度返回


class DotProduct(SimilarityStrategy):
    """
    点积相似度

    TODO: 补全 compute 方法
    """
    # ===== 你的代码 =====
    pass  # TODO: 实现点积计算


class SimilarityComputer:
    """相似度计算上下文"""

    def __init__(self, strategy: SimilarityStrategy):
        self.strategy = strategy

    def compute(self, v1: List[float], v2: List[float]) -> float:
        return self.strategy.compute(v1, v2)


v1, v2 = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
cosine = SimilarityComputer(CosineSimilarity())
euclidean = SimilarityComputer(EuclideanDistance())
dot = SimilarityComputer(DotProduct())

print(f"    余弦相似度: {cosine.compute(v1, v2):.4f}")
print(f"    欧氏相似度: {euclidean.compute(v1, v2):.4f}")
print(f"    点积相似度: {dot.compute(v1, v2):.4f}")


# ── 🚀 RAG 实战：可切换的 Reranker 策略 ──
print("\n  🚀 RAG 实战：根据场景切换 Reranker 策略")
print("  " + "-" * 40)

"""
实战场景：
  你的 Reranker.py 中可能有多种重排策略：
    - NoneReranker    ：不做重排，直接用原始分数
    - RuleReranker    ：基于规则的（如关键词匹配度）
    - CrossEncoderReranker：基于 Cross-Encoder 模型
    - LLMReranker     ：用 LLM 做 Pointwise 评分

  使用策略模式，可以在评测时快速对比不同 reranker 的效果。
"""


class RerankerStrategy(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[dict]) -> List[dict]:
        """
        输入：query + 检索结果列表（含原始分数）
        输出：重排后的结果列表
        """
        pass


class NoRerank(RerankerStrategy):
    """不做重排"""

    def rerank(self, query: str, documents: List[dict]) -> List[dict]:
        return documents


class ScoreBasedRerank(RerankerStrategy):
    """基于原始分数排序（降序）"""

    def rerank(self, query: str, documents: List[dict]) -> List[dict]:
        return sorted(documents, key=lambda d: d.get("score", 0),
                      reverse=True)


class RuleBasedRerank(RerankerStrategy):
    """基于规则的重排：关键词匹配越多，分数越高"""

    def rerank(self, query: str, documents: List[dict]) -> List[dict]:
        query_words = set(query)
        for doc in documents:
            content = doc.get("content", "")
            # 计算关键词命中数作为额外加分
            keyword_bonus = sum(
                1 for w in query_words if w in content)
            doc["score"] = doc.get("score", 0) + keyword_bonus * 0.1
        return sorted(documents, key=lambda d: d.get("score", 0),
                      reverse=True)


class RAGEvaluator:
    """RAG 评测器 —— 使用 Reranker 策略"""

    def __init__(self, reranker: RerankerStrategy):
        self.reranker = reranker

    def set_reranker(self, reranker: RerankerStrategy):
        self.reranker = reranker

    def evaluate_query(self, query: str,
                       raw_docs: List[dict]) -> List[dict]:
        return self.reranker.rerank(query, raw_docs)


evaluator = RAGEvaluator(NoRerank())
sample_docs = [
    {"content": "保研加分政策包括竞赛加分", "score": 0.6},
    {"content": "中南财经大学保研细则", "score": 0.9},
    {"content": "竞赛获奖可以加分", "score": 0.7},
]
print(f"    无重排: {[d['score'] for d in evaluator.evaluate_query('保研加分', sample_docs)]}")
evaluator.set_reranker(RuleBasedRerank())
print(f"    规则重排: {[d['score'] for d in evaluator.evaluate_query('保研加分', sample_docs)]}")


# =============================================================================
# 11. 观察者模式（Observer Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 11：观察者模式（Observer Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：当一个对象的状态发生变化时，如何自动通知所有依赖它的对象？

就像微信公众号：
  你关注了一个公众号（订阅），
  公众号发布新文章时，所有关注者都会收到推送。

场景：
  • 评测过程中，多个 Logger 监听评测事件（控制台 / 文件 / 数据库）
  • GUI 中，Model 变化时自动更新 View
  • 消息队列的发布-订阅模型（Pub/Sub）

核心角色：
  Subject（主题/被观察者）：维护观察者列表，状态变化时通知
  Observer（观察者）：定义更新接口，收到通知后执行自己的逻辑
"""

# ── 完整示例 ──
print("\n  📌 完整示例：评测事件通知系统")
print("  " + "-" * 40)


class EvalObserver(ABC):
    """评测观察者抽象类"""

    @abstractmethod
    def on_eval_start(self, total: int):
        pass

    @abstractmethod
    def on_item_complete(self, index: int, result: dict):
        pass

    @abstractmethod
    def on_eval_end(self, summary: dict):
        pass


class ConsoleLogger(EvalObserver):
    """控制台日志观察者"""

    def on_eval_start(self, total: int):
        print(f"      📺 [Console] 评测开始，共 {total} 条数据")

    def on_item_complete(self, index: int, result: dict):
        print(f"      📺 [Console] 第 {index + 1} 条完成: "
              f"score={result.get('score', 'N/A')}")

    def on_eval_end(self, summary: dict):
        print(f"      📺 [Console] 评测结束，平均分: "
              f"{summary.get('avg_score', 0):.2f}")


class FileLogger(EvalObserver):
    """文件日志观察者"""

    def __init__(self):
        self.buffer = []

    def on_eval_start(self, total: int):
        self.buffer.append(f"评测开始,共{total}条\n")

    def on_item_complete(self, index: int, result: dict):
        self.buffer.append(
            f"第{index + 1}条,score={result.get('score', 'N/A')}\n")

    def on_eval_end(self, summary: dict):
        self.buffer.append(
            f"评测结束,平均分={summary.get('avg_score', 0):.2f}\n")
        # 实际项目中会写入文件
        print(f"      📄 [File] 日志已缓存，共 {len(self.buffer)} 行")


class EvalSubject:
    """评测主题（被观察者）"""

    def __init__(self):
        self._observers: List[EvalObserver] = []

    def attach(self, observer: EvalObserver):
        self._observers.append(observer)

    def detach(self, observer: EvalObserver):
        self._observers.remove(observer)

    def _notify_start(self, total: int):
        for obs in self._observers:
            obs.on_eval_start(total)

    def _notify_item(self, index: int, result: dict):
        for obs in self._observers:
            obs.on_item_complete(index, result)

    def _notify_end(self, summary: dict):
        for obs in self._observers:
            obs.on_eval_end(summary)

    def run_eval(self, data: List[dict]):
        """模拟评测流程"""
        self._notify_start(len(data))
        scores = []
        for i, item in enumerate(data):
            # 模拟打分
            score = sum(ord(c) for c in item.get("question", "")) % 100
            result = {"question": item.get("question"), "score": score}
            scores.append(score)
            time.sleep(0.1)  # 模拟耗时
            self._notify_item(i, result)
        self._notify_end({"avg_score": sum(scores) / len(scores)})


# 运行
subject = EvalSubject()
subject.attach(ConsoleLogger())
subject.attach(FileLogger())
subject.run_eval([
    {"question": "保研加分政策"},
    {"question": "推免细则"},
    {"question": "竞赛加分标准"},
])


# ── 🚀 RAG 实战：评测多路日志输出 ──
print("\n  🚀 RAG 实战：多路日志监听（控制台 + JSON + SQLite）")
print("  " + "-" * 40)


class JSONFileLogger(EvalObserver):
    """JSON 格式日志 — 适合后续分析"""

    def __init__(self, file_path: str = "eval_log.jsonl"):
        self.file_path = file_path
        self.records = []

    def on_eval_start(self, total: int):
        pass

    def on_item_complete(self, index: int, result: dict):
        self.records.append(result)

    def on_eval_end(self, summary: dict):
        # 模拟写入 JSONL
        print(f"      📋 [JSON] 将 {len(self.records)} 条记录写入 "
              f"{self.file_path}（模拟）")
        self.records = []


# 模拟实际评测场景
eval_system = EvalSubject()
eval_system.attach(ConsoleLogger())
eval_system.attach(FileLogger())
eval_system.attach(JSONFileLogger("eval_output.jsonl"))
print("\n    模拟 RAG 评测运行：")
eval_system.run_eval([
    {"question": "问题1", "context": ["文档1"]},
    {"question": "问题2", "context": ["文档2"]},
])
print("    ✅ 观察者模式：一个事件，多个响应！")

# =============================================================================
# 12. 模板方法模式（Template Method Pattern）
# =============================================================================

print("\n" + "─" * 60)
print("模式 12：模板方法模式（Template Method Pattern）")
print("─" * 60)

"""
📖 概念讲解
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题：多个子类有相似的流程，只有部分步骤不同。如何复用流程骨架？

就像做菜的「食谱模板」：
  1. 准备食材  ← 固定
  2. 烹饪      ← 不同菜做法不同（子类实现）
  3. 装盘      ← 固定

场景：
  • 评测流程：加载数据 → 推理 → 打分 → 输出（流程固定，打分方式可变）
  • 数据清洗：读取 → 过滤 → 转换 → 验证 → 写入
  • 模型训练：加载数据 → 预处理 → 训练 → 验证 → 保存

核心思想：
  父类定义算法的骨架（模板方法），
  子类实现具体步骤（钩子方法）。

与策略模式的区别：
  策略：整个算法可替换（多态）
  模板方法：算法骨架固定，部分步骤可替换
"""

# ── 完整示例 ──
print("\n  📌 完整示例：评测流程模板")
print("  " + "-" * 40)


class EvalPipeline(ABC):
    """评测流程模板"""

    def run(self, eval_data_path: str) -> dict:
        """
        模板方法：定义评测的固定流程

        流程步骤：
          1. 加载数据
          2. 对每条数据执行推理
          3. 计算分数
          4. 聚合结果并输出
        """
        print(f"    🔄 评测流程开始...")

        # 步骤 1：加载数据（固定实现）
        data = self.load_data(eval_data_path)
        print(f"    📂 加载了 {len(data)} 条数据")

        # 步骤 2：推理（由子类实现）
        predictions = []
        for item in data:
            pred = self.infer(item)
            predictions.append(pred)

        # 步骤 3：打分（由子类实现）
        scores = self.compute_scores(data, predictions)

        # 步骤 4：输出结果（固定实现，但内容由子类的钩子决定）
        summary = self.build_summary(scores)
        self.output_summary(summary)
        return summary

    def load_data(self, path: str) -> List[dict]:
        """加载评测数据（可被子类重写）"""
        # 模拟加载 JSONL
        return [
            {"question": "测试1", "reference": "答案1"},
            {"question": "测试2", "reference": "答案2"},
        ]

    @abstractmethod
    def infer(self, item: dict) -> dict:
        """推理步骤：由子类实现"""
        pass

    @abstractmethod
    def compute_scores(self, data: List[dict],
                       predictions: List[dict]) -> dict:
        """计算分数：由子类实现"""
        pass

    def build_summary(self, scores: dict) -> dict:
        """构建摘要（可被子类重写的钩子）"""
        return {"scores": scores, "total": sum(
            s.get("correct", 0) for s in scores.values())}

    def output_summary(self, summary: dict):
        """输出结果"""
        print(f"    📊 评测结果: {summary}")


class AccuracyEvalPipeline(EvalPipeline):
    """基于准确率的评测"""

    def infer(self, item: dict) -> dict:
        # 模拟 LLM 推理
        return {"predicted": f"模拟回答:{item['question'][:5]}", "reference": item["reference"]}

    def compute_scores(self, data: List[dict],
                       predictions: List[dict]) -> dict:
        # 返回每条数据的详细结果，供 build_summary 聚合
        per_item = {}
        for i, (d, p) in enumerate(zip(data, predictions)):
            is_correct = d["reference"] in p["predicted"]
            per_item[f"item_{i}"] = {
                "question": d["question"],
                "predicted": p["predicted"],
                "correct": 1 if is_correct else 0,
            }
        return per_item


class MRREvalPipeline(EvalPipeline):
    """基于 MRR 的评测"""

    def infer(self, item: dict) -> dict:
        # 模拟检索排序
        return {"ranked_docs": ["doc_a", "doc_b", "doc_c"], "reference": item["reference"]}

    def compute_scores(self, data: List[dict],
                       predictions: List[dict]) -> dict:
        # 返回每条数据的详细结果
        per_item = {}
        for i, (d, p) in enumerate(zip(data, predictions)):
            # 计算 RR：正确文档在排序中的位置倒数
            rank = (i % 3) + 1  # 模拟排名
            per_item[f"item_{i}"] = {
                "question": d["question"],
                "rr": 1.0 / rank,
                "correct": 1 if rank == 1 else 0,
            }
        return per_item


pipeline_acc = AccuracyEvalPipeline()
pipeline_acc.run("eval.jsonl")

pipeline_mrr = MRREvalPipeline()
pipeline_mrr.run("eval.jsonl")


# ── 🚀 RAG 实战：可扩展的评测框架 ──
print("\n  🚀 RAG 实战：你的 run_eval.py 可以这样重构")
print("  " + "-" * 40)

"""
实战场景：
  你的 run_eval.py 可以基于模板方法模式重构：

  class RAGEvalPipeline(EvalPipeline):
      def load_data(self, path):
          # 复用你现有的 load_jsonl
          pass
  
      def infer(self, item):
          # 调用 RAGFacade.ask() 获取答案
          pass
  
      def compute_scores(self, data, predictions):
          # 复用你 scores.py 中的指标计算
          pass

  好处：
    - 新增评测指标只需继承基类，不用改原有代码（开闭原则）
    - 评测流程统一，减少重复代码
    - 方便做 A/B 测试（不同子类 = 不同评测方案）
"""


# =============================================================================
# 附录 A：设计模式速查表
# =============================================================================

print("\n" + "=" * 60)
print("📋 附录 A：设计模式速查表")
print("=" * 60)

CHEAT_SHEET = """
┌──────────────────┬──────────┬──────────────────────────────────────┬──────────────────────┐
│      模式        │   类型   │              一句话概括                │       RAG 实战        │
├──────────────────┼──────────┼──────────────────────────────────────┼──────────────────────┤
│ 单例 Singleton   │ 创建型   │ 全局只有一个实例                      │ Embedding 模型缓存    │
│ 简单工厂         │ 创建型   │ 一个工厂根据参数创建不同产品           │ LLM 客户端创建        │
│ 工厂方法         │ 创建型   │ 子类决定创建哪个具体产品               │ 检索策略切换          │
│ 抽象工厂         │ 创建型   │ 创建一整套相关的产品族                │ RAG Pipeline 组件族   │
│ 建造者 Builder   │ 创建型   │ 分步构建复杂对象                      │ Prompt 组装           │
├──────────────────┼──────────┼──────────────────────────────────────┼──────────────────────┤
│ 装饰器 Decorator │ 结构型   │ 不修改原代码，动态添加功能             │ @timer @log @cache   │
│ 适配器 Adapter   │ 结构型   │ 让不兼容的接口协同工作                │ 统一 LLM 调用接口     │
│ 代理 Proxy       │ 结构型   │ 控制对对象的访问（限流/缓存/保护）    │ LLM 调用代理          │
│ 外观 Facade      │ 结构型   │ 为复杂子系统提供简单入口              │ RAG 一键问答          │
├──────────────────┼──────────┼──────────────────────────────────────┼──────────────────────┤
│ 策略 Strategy    │ 行为型   │ 算法可互换                            │ Reranker 策略切换     │
│ 观察者 Observer  │ 行为型   │ 一对多通知（发布-订阅）               │ 评测多路日志          │
│ 模板方法 TM      │ 行为型   │ 固定流程骨架，步骤可替换              │ 评测流程模板          │
└──────────────────┴──────────┴──────────────────────────────────────┴──────────────────────┘
"""

print(CHEAT_SHEET)

# =============================================================================
# 附录 B：SOLID 原则简介
# =============================================================================

print("=" * 60)
print("📋 附录 B：SOLID 设计原则")
print("=" * 60)

print("""
SOLID 是面向对象设计的五大原则，设计模式就是这些原则的具体实践：

  S — 单一职责原则 (SRP)
      一个类只负责一件事。
      例子：不要把「检索 + 生成 + 打分」写在一个类里。

  O — 开闭原则 (OCP)
      对扩展开放，对修改关闭。
      例子：工厂方法模式 — 新增检索策略只需加子类，不改原有代码。

  L — 里氏替换原则 (LSP)
      子类必须可以替换父类。
      例子：任何 LLMClient 的子类（OpenAI / Ollama）都能无缝替换。

  I — 接口隔离原则 (ISP)
      不要强迫客户端依赖它不需要的接口。
      例子：把大接口拆成多个小接口。

  D — 依赖倒置原则 (DIP)
      依赖抽象，不要依赖具体实现。
      例子：RAGFacade 依赖 UnifiedLLM（抽象），不依赖 OllamaAdapter（具体）。

记住口诀：「S-O-L-I-D 让代码更 SOLID（坚固）」！
""")

# =============================================================================
# 附录 C：练习测试运行器
# =============================================================================

print("=" * 60)
print("📋 附录 C：练习测试运行器")
print("=" * 60)


def run_all_tests():
    """运行所有练习的测试"""
    passed = 0
    total = 0

    # ── 练习 1：单例装饰器 ──
    total += 1
    try:
        @singleton_exercise
        class TestSingleton:
            def __init__(self, val=0):
                self.val = val

        a = TestSingleton(1)
        b = TestSingleton(2)
        assert a is b, "单例失败：不是同一个实例"
        assert a.val == 1, f"单例失败：val 应该是 1，实际是 {a.val}"
        print("✅ 练习 1 通过（单例装饰器）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 1 失败: {e}")

    # ── 练习 2：LLM 工厂 ──
    total += 1
    try:
        openai_client = LLMFactory.create(
            {"provider": "openai", "api_key": "sk-test",
             "model": "gpt-4"})
        assert isinstance(openai_client, OpenAIClient)

        ollama_client = LLMFactory.create(
            {"provider": "ollama", "model": "qwen3:0.5b"})
        assert isinstance(ollama_client, OllamaClient)

        try:
            LLMFactory.create({"provider": "unknown"})
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass

        print("✅ 练习 2 通过（LLM 工厂）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 2 失败: {e}")

    # ── 练习 3：检索策略工厂 ──
    total += 1
    try:
        vf = VectorSearchFactory()
        assert isinstance(vf.create_search(), VectorSearch)

        kf = KeywordSearchFactory()
        assert isinstance(kf.create_search(), KeywordSearch)

        hf = HybridSearchFactory()
        assert isinstance(hf.create_search(), HybridSearch)
        print("✅ 练习 3 通过（检索策略工厂）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 3 失败: {e}")

    # ── 练习 4：Prompt 建造者 ──
    total += 1
    try:
        prompt = (PromptBuilder()
                  .set_question("测试问题")
                  .add_context("上下文1")
                  .build())
        assert prompt.question == "测试问题"
        assert "上下文1" in prompt.to_string()

        try:
            PromptBuilder().build()  # 没有 question 应该报错
            assert False, "应该抛出 ValueError"
        except ValueError:
            pass

        print("✅ 练习 4 通过（Prompt 建造者）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 4 失败: {e}")

    # ── 练习 5：缓存装饰器 ──
    total += 1
    try:
        call_count = [0]

        @cache
        def test_func(x, y=1):
            call_count[0] += 1
            return x + y

        assert test_func(1, 2) == 3
        assert test_func(1, 2) == 3
        assert call_count[0] == 1, "缓存未生效"
        assert test_func(2, 3) == 5
        assert call_count[0] == 2
        print("✅ 练习 5 通过（缓存装饰器）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 5 失败: {e}")

    # ── 练习 6：Ollama 适配器 ──
    total += 1
    try:
        adapter = OllamaAdapter("qwen3:0.5b")
        assert isinstance(adapter, UnifiedLLM)
        result = adapter.chat("测试")
        assert "Ollama" in result, f"结果中应包含 Ollama，实际: {result}"
        print("✅ 练习 6 通过（Ollama 适配器）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 6 失败: {e}")

    # ── 练习 7：相似度策略 ──
    total += 1
    try:
        euclidean = EuclideanDistance()
        result = euclidean.compute([1, 2], [4, 5])
        assert 0 < result <= 1, f"欧氏相似度应在 (0,1] 之间，实际: {result}"

        dot = DotProduct()
        result = dot.compute([1, 2, 3], [4, 5, 6])
        assert result == 32, f"点积应为 32，实际: {result}"
        print("✅ 练习 7 通过（相似度策略）")
        passed += 1
    except Exception as e:
        print(f"❌ 练习 7 失败: {e}")

    # ── 总结 ──
    print(f"\n{'=' * 50}")
    print(f"🎯 练习结果: {passed}/{total} 通过")
    if passed == total:
        print("🎉 恭喜！全部练习通过！")
        print("\n📚 下一步建议：")
        print("   1. 试着在你的 RAG 项目中应用这些模式")
        print("   2. 阅读《精通 Python 设计模式》深入理解")
        print("   3. 在 LeetCode 刷题时思考用了哪些设计模式")
        print("   4. 阅读优秀开源项目（如 Django/Flask）的源码")
    else:
        print(f"💪 还有 {total - passed} 个练习未通过，继续加油！")
        print("   可以滚动到对应练习的注释中查看提示。")


if __name__ == "__main__":
    run_all_tests()