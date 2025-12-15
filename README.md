# rag-med-assistant

## 1.项目摘要
本项目旨在解决大型语言模型在专业医疗领域知识的准确性和时效性问题，通过构建一套基于检索增强生成 (RAG) 架构的智能问答系统。系统专注于提供肝胆胰外科手术患者术后营养支持和康复指导。通过整合权威医疗指南和文献，本项目确保为患者和初级医护人员提供快速、准确、且基于最新证据的营养管理建议，从而辅助术后康复过程，提高患者依从性和治疗效果。
### 核心目标:
* 提供基于最新医疗证据的专业营养指导。
* 实现 95% 以上的答案忠实度 (Faithfulness)。
* 建立可增量更新的专业知识库。

## 2.技术栈
| 模块 | 技术 / 库 | 描述 |
| :--- | :--- | :--- |
| **编程语言** | Python 3.10+ | 项目主要开发语言。 |
| **框架** | FastAPI | 用于构建高性能的检索和 RAG API 服务。 |
| **RAG 编排** | LangChain  | 负责整合检索、提示词工程和 LLM 调用的核心逻辑。 |
| **向量数据库** | ChromaDB  | 轻量级、易于部署的向量存储解决方案。支持本地存储和 Chroma Cloud（目前还没实现）。 |
| **LLM / Embedding** | Deepseek / Llama-3 (或 OpenAI API) | 核心生成模型和文本转向量模型。 |
| **环境管理** | uv | 依赖和虚拟环境管理。 |

## 3.快速开始

### 3.1 环境要求
- Python 3.13 或更高版本（项目配置要求，见 `pyproject.toml`）
- OpenAI API Key（用于 LLM 和 Embedding）

### 3.2 安装依赖

项目依赖包括：
- `chromadb` - 向量数据库（支持本地和云端）
- `langchain` - RAG 框架
- `langchain-openai` - OpenAI 集成
- `fastapi` / `uvicorn` - Web API 框架
- `python-dotenv` - 环境变量管理

#### 方式一：使用 uv（推荐）

**1. 安装 uv**

macOS/Linux:
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```



或者使用 pip 安装：
```bash
pip install uv
```

**2. 创建虚拟环境**

```bash
uv venv
```

这会在项目目录下创建 `.venv` 虚拟环境。

**3. 激活虚拟环境**

macOS/Linux:
```bash
source .venv/bin/activate
```



**4. 安装项目依赖**

```bash
uv sync
```

#### 方式二：使用 pip
```zsh
pip install -r requirements.txt
```



### 3.3 配置环境变量

#### 方式一：使用本地向量数据库

创建 `.env` 文件：

```zsh
# 必需：OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here
```

系统会自动使用本地向量数据库（存储在 `./chroma_db_medical` 目录）。

#### 方式二：使用 Chroma Cloud （没实现，有点问题，应该是没充钱数据存放不了）

1. **注册 Chroma Cloud 账号**
   - 访问 [Chroma Cloud](https://www.trychroma.com/)
   - 注册账号并创建项目
   - 获取 API Key、Tenant ID 等信息

2. **创建 `.env` 文件**：

```bash
# 必需：OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Chroma Cloud 配置（可选，如果设置则使用云端数据库）
CHROMA_API_KEY=your_chroma_api_key_here
CHROMA_TENANT=your_tenant_id
CHROMA_DATABASE=your_database_name
CHROMA_COLLECTION=medical_qa
```

**注意**：
- 如果不设置 `CHROMA_API_KEY`，系统会自动使用本地向量数据库
- 如果设置了 `CHROMA_API_KEY` 但连接失败，系统会自动回退到本地数据库
- 详细配置说明请参考 `CHROMA_CLOUD_SETUP.md`

### 3.4 数据导入

首次使用前，需要将医疗问答数据导入向量数据库：

```bash
python src/ingest.py
```

该脚本会：
1. 读取 `QA_V0.md` 文件（格式：`Q：... A：...`）
2. 解析问答对
3. 生成向量嵌入
4. 存储到 ChromaDB（本地或云端）



### 3.5 启动服务


```zsh
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

服务启动后，访问：
- API 文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000

### 3.6 启动前端界面



在项目根目录运行：

```zsh
python start_frontend.py
```

前端服务器将在 `http://localhost:8080` 启动，在浏览器中访问该地址即可使用。




## 4.API 使用示例

### 4.1 发送问题

```zsh
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"question": "术后第一天可以吃什么？"}'
```

### 4.2 响应格式

```json
{
  "answer": "根据术后营养管理指南...",
  "status": "success"
}
```

## 5.项目结构

```
rag-med-assistant/
├── src/
│   ├── api.py          # FastAPI 服务入口
│   ├── rag.py          # RAG 链构建和命令行交互
│   └── ingest.py       # 数据导入脚本
├── frontend/
│   └── index.html      # Web 前端界面
├── chroma_db_medical/  # 本地向量数据库（自动生成）
├── QA_V0.md           # 医疗问答数据源
├── requirements.txt    # Python 依赖
├── pyproject.toml      # 项目配置（uv/poetry）
└── README.md          # 本文件
```



