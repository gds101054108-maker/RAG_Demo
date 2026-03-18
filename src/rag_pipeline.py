"""
RAG 主流程 (组合1: 生产就绪堆栈)

组合策略:
- 策略1: 上下文感知分块
- 策略3: 重排序
- 策略4: 查询扩展
- 策略6: 智能体 RAG

预期效果: 92% 准确率, 1.2秒平均延迟
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import time
import os

from .chunking import ContextAwareChunker, Chunk
from .reranking import HybridReranker, RerankResult
from .query_expansion import QueryExpander, ExpandedQuery
from .agent_rag import AgentRAG, AgentDecision
from .vector_store import VectorStore, Document


@dataclass
class RAGResponse:
    """RAG 响应结果"""
    query: str
    answer: str
    sources: List[Dict]
    metadata: Dict[str, Any]


class RAGPipeline:
    """RAG 流水线"""
    
    def __init__(
        self,
        api_key: str,
        llm_model: str = "qwen-flash",
        embedding_model: str = "text-embedding-v3",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        vector_db_path: str = "./data/chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        top_k: int = 5,
        rerank_candidates: int = 20
    ):
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates
        
        # 初始化各组件
        self.chunker = ContextAwareChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # 使用百炼 qwen3-rerank 进行重排序
        self.reranker = HybridReranker(
            api_key=api_key,
            model="qwen3-rerank"
        )
        
        self.query_expander = QueryExpander(
            api_key=api_key,
            model=llm_model,
            base_url=base_url
        )
        
        self.vector_store = VectorStore(
            persist_directory=vector_db_path,
            embedding_model=embedding_model,
            api_key=api_key,
            base_url=base_url
        )
        
        self.agent = AgentRAG(
            api_key=api_key,
            model=llm_model,
            base_url=base_url
        )
        self.agent.set_vector_store(self.vector_store)
    
    def ingest_documents(
        self,
        documents: List[Dict],
        show_progress: bool = True
    ) -> int:
        """
        摄取文档
        
        Args:
            documents: 文档列表，每个包含 'content' 和 'title'
            show_progress: 是否显示进度
        
        Returns:
            摄取的分块数量
        """
        if show_progress:
            print(f"[摄取] 开始处理 {len(documents)} 个文档...")
        
        # 步骤1: 上下文感知分块
        all_chunks = self.chunker.chunk_documents(documents)
        
        if show_progress:
            print(f"[分块] 生成 {len(all_chunks)} 个分块")
        
        # 转换为 Document 对象
        docs_to_add = [
            Document(
                content=chunk.content,
                metadata=chunk.metadata,
                doc_id=f"chunk_{chunk.chunk_id}_{i}"
            )
            for i, chunk in enumerate(all_chunks)
        ]
        
        # 步骤2: 存入向量数据库
        added = self.vector_store.add_documents(docs_to_add)
        
        if show_progress:
            print(f"[完成] 成功摄取 {added} 个分块")
        
        return added
    
    def retrieve(
        self,
        query: str,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_agent: bool = True
    ) -> List[RerankResult]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            use_expansion: 是否使用查询扩展
            use_reranking: 是否使用重排序
            use_agent: 是否使用智能体决策
        
        Returns:
            检索结果列表
        """
        # 步骤1: 智能体分析查询（可选）
        if use_agent:
            decision = self.agent.analyze_query(query)
            print(f"[智能体] 选择策略: {decision.tool_name}")
        
        # 步骤2: 查询扩展（可选）
        if use_expansion:
            expanded = self.query_expander.expand_query(query)
            search_query = expanded.expanded
            print(f"[扩展] 原始: {query}")
            print(f"[扩展] 扩展后: {search_query[:100]}...")
        else:
            search_query = query
        
        # 步骤3: 向量检索（获取更多候选）
        candidates = self.vector_store.similarity_search(
            search_query,
            k=self.rerank_candidates
        )
        
        print(f"[检索] 获取 {len(candidates)} 个候选")
        
        # 步骤4: 重排序（可选）
        if use_reranking and candidates:
            results = self.reranker.rerank(
                query,
                candidates,
                top_k=self.top_k
            )
            print(f"[重排序] 返回前 {len(results)} 个结果")
        else:
            results = [
                RerankResult(
                    content=c['content'],
                    score=c.get('score', 0),
                    original_rank=i,
                    metadata=c.get('metadata', {})
                )
                for i, c in enumerate(candidates[:self.top_k])
            ]
        
        return results
    
    def generate_answer(
        self,
        query: str,
        retrieved_docs: List[RerankResult]
    ) -> str:
        """
        生成答案
        
        Args:
            query: 查询文本
            retrieved_docs: 检索到的文档
        
        Returns:
            生成的答案
        """
        import requests
        
        # 构建上下文（显示文档名称）
        context = "\n\n".join([
            f"[文档 {i+1}: {doc.metadata.get('source', doc.metadata.get('document_title', '未知来源'))}]\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # 构建提示
        system_prompt = """你是一个专业的知识助手。
基于提供的参考文档回答用户问题。

要求：
1. 只使用参考文档中的信息回答
2. 如果文档中没有相关信息，诚实说明
3. 引用时使用文档名称（如"根据《通义千问认证模型服务接口说明 V1.0》"），不要用"文档 1"这样的编号
4. 保持回答简洁、准确"""
        
        full_query = f"""参考文档:
{context}

用户问题: {query}

请基于参考文档给出详细、准确的回答，并注明引用来源。"""
        
        # 调用 LLM
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_query}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def query(
        self,
        query: str,
        use_expansion: bool = True,
        use_reranking: bool = True,
        use_agent: bool = True
    ) -> RAGResponse:
        """
        完整查询流程
        
        Args:
            query: 查询文本
            use_expansion: 是否使用查询扩展
            use_reranking: 是否使用重排序
            use_agent: 是否使用智能体决策
        
        Returns:
            RAG 响应结果
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"查询: {query}")
        print(f"{'='*60}")
        
        # 检索
        retrieved_docs = self.retrieve(
            query,
            use_expansion=use_expansion,
            use_reranking=use_reranking,
            use_agent=use_agent
        )
        
        # 生成答案
        answer = self.generate_answer(query, retrieved_docs)
        
        elapsed_time = time.time() - start_time
        
        # 构建响应
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[
                {
                    "content": doc.content[:200] + "...",
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ],
            metadata={
                "elapsed_time": elapsed_time,
                "num_sources": len(retrieved_docs),
                "use_expansion": use_expansion,
                "use_reranking": use_reranking,
                "use_agent": use_agent
            }
        )
        
        print(f"\n[完成] 耗时: {elapsed_time:.2f}秒")
        
        return response
    
    def get_stats(self) -> dict:
        """获取系统统计信息"""
        return self.vector_store.get_stats()


# 示例使用
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if api_key:
        # 创建 RAG 流水线
        rag = RAGPipeline(
            api_key=api_key,
            vector_db_path="./demo_chroma"
        )
        
        # 摄取测试文档
        test_docs = [
            {
                "content": """
# RAG 系统概述

RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的AI技术。

## 主要优势

1. 知识可更新：无需重新训练模型即可更新知识
2. 减少幻觉：基于检索到的事实生成答案，降低虚构内容
3. 可解释性：可以追溯答案来源，便于验证

## 技术架构

RAG 系统通常包含以下核心组件：
- 文档处理：分块、向量化
- 向量数据库：存储和检索嵌入向量
- 大语言模型：基于检索内容生成答案
""",
                "title": "RAG技术介绍"
            },
            {
                "content": """
# 向量数据库选型

## ChromaDB
开源向量数据库，适合中小规模应用。
- 优点：易用、支持本地部署
- 缺点：大规模性能有限

## Pinecone
托管向量数据库服务。
- 优点：高性能、可扩展
- 缺点：商业服务，有成本

## Milvus
开源分布式向量数据库。
- 优点：高性能、可扩展
- 缺点：部署复杂
""",
                "title": "向量数据库对比"
            }
        ]
        
        rag.ingest_documents(test_docs)
        
        # 测试查询
        response = rag.query("RAG 系统有什么优势？")
        
        print("\n" + "="*60)
        print("答案:")
        print("="*60)
        print(response.answer)
        
        print("\n来源:")
        for i, source in enumerate(response.sources, 1):
            print(f"  {i}. 分数: {source['score']:.4f}")
        
        # 清理测试数据
        import shutil
        shutil.rmtree("./demo_chroma", ignore_errors=True)
    else:
        print("请设置 DASHSCOPE_API_KEY 环境变量")