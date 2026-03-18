"""
策略3: 重排序 (Reranking)

使用阿里云百炼 qwen3-rerank 模型进行重排序。
两阶段检索：
1. 快速向量检索找到候选（召回更多候选）
2. qwen3-rerank 重新评分，提高精度

向量相似度 ≠ 语义相关性，重排序可以修复向量搜索的错误。
"""

from typing import List, Optional
from dataclasses import dataclass
import requests


@dataclass
class RerankResult:
    """重排序结果"""
    content: str
    score: float
    original_rank: int
    metadata: dict


class QwenReranker:
    """使用阿里云百炼 qwen3-rerank 的重排序器"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank"
    ):
        self.api_key = api_key
        self.model = model
        # qwen3-rerank 使用 OpenAI 兼容 API
        self.api_url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    
    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        使用 qwen3-rerank 对候选文档进行重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表，每个包含 'content', 'score', 'metadata'
            top_k: 返回前 K 个结果
        
        Returns:
            重排序后的结果列表
        """
        if not candidates:
            return []
        
        # 准备文档列表
        documents = [c['content'] for c in candidates]
        
        # 调用百炼 rerank API (OpenAI 兼容格式)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
            "instruct": "Given a web search query, retrieve relevant passages that answer the query."
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[重排序] API 错误: {response.status_code} - {response.text}")
                # 返回原始排序
                return self._fallback_rerank(candidates, top_k)
            
            result = response.json()
            
            # 解析结果
            reranked_results = []
            for item in result.get('results', []):
                index = item.get('index', 0)
                score = item.get('relevance_score', item.get('score', 0))
                
                candidate = candidates[index]
                reranked_results.append(RerankResult(
                    content=candidate['content'],
                    score=score,
                    original_rank=index,
                    metadata=candidate.get('metadata', {})
                ))
            
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"[重排序] 调用失败: {e}")
            return self._fallback_rerank(candidates, top_k)
    
    def _fallback_rerank(
        self,
        candidates: List[dict],
        top_k: int
    ) -> List[RerankResult]:
        """后备排序方法：使用原始分数"""
        results = []
        for i, candidate in enumerate(candidates[:top_k]):
            results.append(RerankResult(
                content=candidate['content'],
                score=candidate.get('score', 0),
                original_rank=i,
                metadata=candidate.get('metadata', {})
            ))
        return results


class HybridReranker:
    """混合重排序器：结合 qwen3-rerank 和关键词匹配"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        rerank_weight: float = 0.8,
        keyword_weight: float = 0.2
    ):
        self.rerank_weight = rerank_weight
        self.keyword_weight = keyword_weight
        self.qwen_reranker = QwenReranker(
            api_key=api_key,
            model=model
        )
    
    def _compute_keyword_score(self, query: str, content: str) -> float:
        """计算关键词匹配分数"""
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        if not query_terms:
            return 0.0
        
        # 关键词覆盖率
        matched = sum(1 for term in query_terms if term in content_lower)
        coverage = matched / len(query_terms)
        
        # 关键词出现频率
        freq_score = sum(content_lower.count(term) for term in query_terms) / 100
        
        return min(coverage + freq_score, 1.0)
    
    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: int = 5
    ) -> List[RerankResult]:
        """
        混合重排序
        
        Args:
            query: 查询文本
            candidates: 候选文档列表
            top_k: 返回结果数量
        
        Returns:
            重排序结果列表
        """
        if not candidates:
            return []
        
        # 使用 qwen3-rerank 进行语义重排序
        reranked = self.qwen_reranker.rerank(query, candidates, top_k * 2)
        
        # 结合关键词分数
        final_results = []
        for result in reranked:
            keyword_score = self._compute_keyword_score(query, result.content)
            
            # 混合分数
            final_score = (
                self.rerank_weight * result.score +
                self.keyword_weight * keyword_score
            )
            
            final_results.append(RerankResult(
                content=result.content,
                score=final_score,
                original_rank=result.original_rank,
                metadata=result.metadata
            ))
        
        # 最终排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]


# 示例使用
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if api_key:
        reranker = QwenReranker(api_key=api_key)
        
        query = "什么是 RAG 系统的主要优势？"
        
        candidates = [
            {
                "content": "RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术，主要优势包括知识可更新、减少幻觉。",
                "score": 0.82,
                "metadata": {"source": "doc1"}
            },
            {
                "content": "机器学习模型训练需要大量数据和计算资源。",
                "score": 0.78,
                "metadata": {"source": "doc2"}
            },
            {
                "content": "RAG 系统的主要优势：1. 知识可更新 2. 减少幻觉 3. 可解释性强。",
                "score": 0.75,
                "metadata": {"source": "doc3"}
            },
        ]
        
        results = reranker.rerank(query, candidates, top_k=3)
        
        print("qwen3-rerank 重排序结果：")
        for i, result in enumerate(results):
            print(f"\n排名 {i+1}:")
            print(f"分数: {result.score:.4f}")
            print(f"原始排名: {result.original_rank + 1}")
            print(f"内容: {result.content[:80]}...")
    else:
        print("请设置 DASHSCOPE_API_KEY 环境变量")