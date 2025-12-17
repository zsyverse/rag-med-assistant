import os
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

SOURCE_FILE = "QA_V0.md"       # 数据源文件路径
DB_PATH = "./chroma_db_medical" # 向量数据库在本地硬盘的存储文件夹