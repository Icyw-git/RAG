from VectorBase import VectorStore
from utils import ReadFiles
from LLM import OpenAIChat
from Embeddings import OpenAIEmbedding

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

source,chunk_id,content,scores = vector.query(question, EmbeddingModel=embedding, k=3)

assert len(content) == len(scores)
context=''
for i,chunk in enumerate(content):
    context+=f'\n\n====chunk {i},score:{scores[i]:.4f},source:{source[i]},chunk_id:{chunk_id[i]}====\n{chunk}\n'

best_score=scores[0] #增加拒答机制，如果最高的相似度分数低于某个阈值，就拒绝回答，提示用户数据库中没有足够的相关内容。
threshold=0.35
if best_score<threshold:
    print("数据库中没有足够的相关内容，我不确定。你可以尝试调整一下问题，或者增加数据库中的内容。")
else:
    chat = OpenAIChat(model='Qwen/Qwen2.5-32B-Instruct')
    print(chat.chat(question, [], context))

    print('Retrieved top-k chunks:',context)

    print(best_score)



