import os
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

SOURCE_FILE = "QA_V0.md"       # 数据源文件路径
DB_PATH = "./chroma_db_medical" # 向量数据库在本地硬盘的存储文件夹

def parse_qa_file(file_path):
    # ... (读取文件内容到 content 变量) ...

    # 🛠️ 核心正则表达式
    # r"..." : Python 的原始字符串，避免转义符干扰
    # Q：(.*?) : 匹配以 "Q：" 开头的内容，(.*?) 是非贪婪匹配，只匹配到下一个条件之前
    # \n+A：   : 匹配一个或多个换行符后紧接 "A："
    # (.*?)    : 匹配答案内容
    # (?=\n+Q：|$) : 正向预查（Lookahead）。意思是“匹配直到遇到下一个 'Q：' 或者文件结束为止”。
    #                这非常重要，它防止了一个 Q 匹配了整个文件剩下的所有内容。
    pattern = r"Q：(.*?)\n+A：(.*?)(?=\n+Q：|$)"
    
    # re.DOTALL : 让 '.' 符号可以匹配换行符。因为答案通常是多行的，不加这个参数只能匹配单行。
    matches = re.findall(pattern, content, re.DOTALL)
    
    # ... (将匹配结果整理成字典列表) ...
    return qa_pairs