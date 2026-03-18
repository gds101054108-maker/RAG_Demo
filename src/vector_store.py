"""
向量存储模块

使用 ChromaDB 作为向量数据库，支持：
- 文档存储和检索
- 语义搜索
- 关键词搜索
- 元数据过滤
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import os
import json


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: dict
    doc_id: Optional[str] = None


class VectorStore:
    """向量存储管理器"""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "rag_documents",
        embedding_model: str = "text-embedding-v3",
        api_key: str = "",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.api_key = api_key
        self.base_url = base_url
        
        self.client = None
        self.collection = None
        
        # 延迟初始化
        self._initialized = False
    
    def _ensure_initialized(self):
        """确保数据库已初始化"""
        if not self._initialized:
            self._init_chroma()
            self._initialized = True
    
    def _init_chroma(self):
        """初始化 ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # 创建持久化目录
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # 初始化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"[向量存储] 初始化完成，集合: {self.collection_name}")
        except Exception as e:
            print(f"[向量存储] 初始化失败: {e}")
            raise
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """获取文本嵌入向量"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        
        # 批量处理，阿里云限制每次最多 10 个文本
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # 阿里云百炼 API 格式
            payload = {
                "model": self.embedding_model,
                "input": batch
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code != 200:
                    print(f"[嵌入] API 错误: {response.status_code} - {response.text}")
                    raise Exception(f"API 调用失败: {response.status_code}")
                
                result = response.json()
                
                # 按索引排序获取嵌入
                batch_embeddings = [None] * len(batch)
                for item in result['data']:
                    idx = item['index']
                    batch_embeddings[idx] = item['embedding']
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"[嵌入] 获取向量失败: {e}")
                raise
        
        return all_embeddings
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> int:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
        
        Returns:
            添加的文档数量
        """
        self._ensure_initialized()
        
        if not documents:
            return 0
        
        total_added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 准备数据
            ids = [doc.doc_id or f"doc_{i + j}" for j, doc in enumerate(batch)]
            contents = [doc.content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            # 获取嵌入向量
            embeddings = self._get_embeddings(contents)
            
            # 添加到集合
            self.collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            total_added += len(batch)
            print(f"[向量存储] 已添加 {total_added}/{len(documents)} 个文档")
        
        return total_added
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None
    ) -> List[Dict]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
        
        Returns:
            搜索结果列表
        """
        self._ensure_initialized()
        
        # 获取查询向量
        query_embedding = self._get_embeddings([query])[0]
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter
        )
        
        # 格式化结果
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # 转换为相似度
                'id': results['ids'][0][i]
            })
        
        return documents
    
    def keyword_search(
        self,
        keyword: str,
        k: int = 5
    ) -> List[Dict]:
        """
        关键词搜索
        
        Args:
            keyword: 关键词
            k: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        self._ensure_initialized()
        
        # 使用 ChromaDB 的 where_document 过滤
        results = self.collection.query(
            query_texts=[keyword],
            n_results=k,
            where_document={"$contains": keyword}
        )
        
        # 格式化结果
        documents = []
        for i in range(len(results['ids'][0])):
            documents.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'id': results['ids'][0][i]
            })
        
        return documents
    
    def search_by_metadata(
        self,
        metadata_filter: dict,
        limit: int = 100
    ) -> List[Dict]:
        """
        根据元数据搜索
        
        Args:
            metadata_filter: 元数据过滤条件
            limit: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        self._ensure_initialized()
        
        # 获取所有匹配的文档
        results = self.collection.get(
            where=metadata_filter,
            limit=limit
        )
        
        # 格式化结果
        documents = []
        for i in range(len(results['ids'])):
            documents.append({
                'content': results['documents'][i],
                'metadata': results['metadatas'][i],
                'id': results['ids'][i]
            })
        
        return documents
    
    def delete_documents(self, ids: List[str]):
        """删除文档"""
        self._ensure_initialized()
        self.collection.delete(ids=ids)
    
    def clear_collection(self):
        """清空集合"""
        self._ensure_initialized()
        # 删除并重新创建集合
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_stats(self) -> dict:
        """获取存储统计信息"""
        self._ensure_initialized()
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


# 示例使用
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if api_key:
        store = VectorStore(
            persist_directory="./test_chroma",
            api_key=api_key
        )
        
        # 添加测试文档
        docs = [
            Document(
                content="RAG 是检索增强生成技术，结合了信息检索和文本生成。",
                metadata={"topic": "RAG", "source": "intro"},
                doc_id="doc_1"
            ),
            Document(
                content="向量数据库用于存储文档的嵌入表示，支持高效相似度搜索。",
                metadata={"topic": "向量数据库", "source": "tech"},
                doc_id="doc_2"
            )
        ]
        
        store.add_documents(docs)
        
        # 搜索测试
        results = store.similarity_search("什么是 RAG？", k=2)
        
        print("\n搜索结果:")
        for r in results:
            print(f"内容: {r['content'][:50]}...")
            print(f"分数: {r['score']:.4f}")
        
        # 清理测试数据
        import shutil
        shutil.rmtree("./test_chroma", ignore_errors=True)
    else:
        print("请设置 DASHSCOPE_API_KEY 环境变量")