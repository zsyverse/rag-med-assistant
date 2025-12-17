import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ================= 配置 =================
DB_PATH = "./chroma_db_medical"
# =======================================

def get_rag_chain():
    # 1. 检查数据库是否存在
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("❌ 向量数据库未找到，请先运行 src/ingest_md.py")

    # 2. 加载数据库
    embedding_model = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    
    # 3. 创建检索器
    # k=3: 每次找 3 条最相关的 QA 给大模型参考
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. 初始化大模型
    # temperature=0: 医疗建议必须严谨，不要发散
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 5. 定义 Prompt (角色设定)
    template = """
    你是一名专业的肝胆胰外科术后康复助手。
    请基于以下【已知信息】回答用户的问题。
    
    要求：
    1. 语气专业、亲切、富有同理心。
    2. 如果【已知信息】中没有答案，请明确告知“资料库中没有相关信息，建议咨询主治医生”，严禁瞎编。
    3. 答案要条理清晰。

    【已知信息】：
    {context}

    用户问题：{question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # 6. 构造链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def start_chat():
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ 错误: 请先设置环境变量 OPENAI_API_KEY")
        return

    print("🏥 肝胆胰术后康复助手正在启动...")
    try:
        chain = get_rag_chain()
        print("✅ 系统就绪！请输入问题 (输入 'q' 退出)")
        print("-" * 50)
        
        while True:
            query = input("\n👤 患者提问: ")
            if query.lower() in ['q', 'quit', 'exit']:
                break
            
            if not query.strip():
                continue

            print("🤖 正在思考...", end="", flush=True)
            start_time = time.time()
            
            # 调用链
            response = chain.invoke(query)
            
            end_time = time.time()
            print(f"\r🤖 助手回答 ({end_time - start_time:.2f}s):")
            print(response)
            print("-" * 50)

    except Exception as e:
        print(f"\n❌ 系统错误: {e}")

if __name__ == "__main__":
    start_chat()