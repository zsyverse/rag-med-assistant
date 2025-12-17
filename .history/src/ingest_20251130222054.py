import os
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# ================= 配置 =================
SOURCE_FILE = "QA_V0.md"       # 您上传的文件名
DB_PATH = "./chroma_db_medical" # 向量数据库存储路径
# =======================================

def parse_qa_file(file_path):
    """
    解析 Q：... A：... 格式的 Markdown 文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式提取 QA 对
    # 逻辑：匹配 "Q：" 开头，中间是问题，然后 "A：" 开头，中间是答案
    # (?s) 开启 DOTALL 模式，让 . 可以匹配换行符
    # (?:Q：|$)用于非捕获组，匹配下一个 Q 的开始或文件结束
    pattern = r"Q：(.*?)\n+A：(.*?)(?=\n+Q：|$)"
    
    matches = re.findall(pattern, content, re.DOTALL)
    
    qa_pairs = []
    for q, a in matches:
        qa_pairs.append({
            "question": q.strip(),
            "answer": a.strip()
        })
    
    return qa_pairs

def ingest_data():
    if not os.path.exists(SOURCE_FILE):
        print(f"❌ 错误: 找不到文件 {SOURCE_FILE}")
        return

    print(f"📖 正在解析文件: {SOURCE_FILE} ...")
    qa_list = parse_qa_file(SOURCE_FILE)
    
    if not qa_list:
        print("⚠️ 未找到有效的 QA 对，请检查文件格式是否为 'Q：... A：...'")
        return

    print(f"✅ 成功解析出 {len(qa_list)} 条问答数据。")
    print(f"   示例: Q: {qa_list[0]['question'][:20]}...")

    # 转换 LangChain Document
    documents = []
    for item in qa_list:
        # 核心策略：
        # 内容 (page_content): 拼接 Q 和 A，这样检索时既能匹配问题关键词，也能匹配答案关键词
        # 也可以尝试只存 Question 到 page_content，把 Answer 存到 metadata (适合精确匹配)
        # 这里我们采用通用的 Q+A 拼接法
        content = f"问题：{item['question']}\n答案：{item['answer']}"
        
        metadata = {
            "source": SOURCE_FILE,
            "original_q": item['question']
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    # 向量化并存储
    print("🔄 正在生成向量并存入 ChromaDB (这可能需要一点时间)...")
    
    # 确保设置了 OPENAI_API_KEY
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ 错误: 请先设置环境变量 OPENAI_API_KEY")
        return

    embedding_model = OpenAIEmbeddings()
    
    # 创建并持久化数据库
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    
    print(f"🎉 入库成功！数据已保存到 {DB_PATH}")

if __name__ == "__main__":
    ingest_data()
