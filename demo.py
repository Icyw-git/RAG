from VectorBase import VectorStore  #导入vectorstore模块 即储存文本信息的向量数据库模块
from utils import ReadFiles  #导入utils模块中的ReadFiles类，ReadFiles类用于读取文件内容并进行分割处理
from LLM import OpenAIChat #导入llm模块中的openaichat模块，调用api接口进行问答
from Embeddings import OpenAIEmbedding #导入embeddding模块中的OpenAIEmbedding类，调用api接口将文本转换成向量表示，在初次导入和后续添加文本时都需要使用embedding模型将文本转换成向量表示，以便存储在vectorstore中，并在查询时计算相似度分数。

# 没有保存数据库
docs = ReadFiles('./data').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = OpenAIEmbedding() # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
vector.persist(path='storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库


# vector=VectorStore() # 创建一个新的VectorStore对象
# embedding=OpenAIEmbedding()
# vector.load_vector('./storage') # 加载本地的数据库

question ='推免具体要求包括什么？'

source,chunk_id,content,scores = vector.query(question, EmbeddingModel=embedding, k=3) #query函数返回与问题最相关的top_k条数据，还有相关性分数，以及对应的source和chunk_id等信息，source和chunk_id可以用来在回答中给出引用，content是实际的文本内容，scores是相似度分数，可以用来判断检索结果的相关程度。

assert len(content) == len(scores)
context=''
for i,chunk in enumerate(content):
    context+=f'\n\n====chunk {i},score:{scores[i]:.4f},source:{source[i]},chunk_id:{chunk_id[i]}====\n{chunk}\n' #将检索到的文本内容按照一定的格式拼接成一个字符串，作为回答问题的上下文信息，格式包括chunk的编号、相似度分数、source和chunk_id等信息，方便后续在回答中引用和说明。

best_score=scores[0] #增加拒答机制，如果最高的相似度分数低于某个阈值，就拒绝回答，提示用户数据库中没有足够的相关内容。
threshold=0.35
if best_score<threshold:
    print("数据库中没有足够的相关内容，我不确定。你可以尝试调整一下问题，或者增加数据库中的内容。")
else:
    chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
    print(chat.chat(question, [], context))

    print('Retrieved top-k chunks:',context)

    print(best_score)



