import os
import sys
from pathlib import Path
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

load_dotenv()

from src.rag import get_rag_chain

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    status: str = "success"

app = FastAPI(
    title="肝胆胰康复助手 API",
    description="基于 RAG 的智能问答服务",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain = None

@app.get("/")
async def root():
    return {
        "message": "肝胆胰康复助手 API",
        "version": "1.0",
        "docs": "/docs",
        "chat_endpoint": "/chat"
    }

@app.on_event("startup")
async def startup_event():
    global rag_chain
    try:
        if "OPENAI_API_KEY" not in os.environ:
            print("⚠️  警告: 未设置 OPENAI_API_KEY")
        
        rag_chain = get_rag_chain()
        print("✅ RAG 模型链加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
    
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=500, detail="模型未初始化")
    
    try:
        answer = rag_chain.invoke(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        print(f"推理错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)