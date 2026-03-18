"""
RAG 系统配置管理
"""
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# 加载环境变量
load_dotenv()


class Settings(BaseSettings):
    """系统配置"""
    
    # API 配置
    dashscope_api_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    
    # 模型配置
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
    llm_model: str = os.getenv("LLM_MODEL", "qwen-flash")
    
    # RAG 配置
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "50"))
    
    # 路径配置
    data_dir: str = "data/documents"
    vector_db_dir: str = "data/chroma_db"
    
    @property
    def dashscope_base_url(self) -> str:
        """阿里云百炼 API 基础 URL"""
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"


# 全局配置实例
settings = Settings()