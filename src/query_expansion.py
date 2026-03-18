"""
策略4: 查询扩展 (Query Expansion)

使用 LLM 将简短查询扩展为更详细、全面的版本。
处理模糊查询，提高检索精度。
"""

from typing import List, Optional
from dataclasses import dataclass
import requests
import json


@dataclass
class ExpandedQuery:
    """扩展查询结果"""
    original: str
    expanded: str
    variations: List[str]


class QueryExpander:
    """查询扩展器"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
    
    def _call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ) -> str:
        """调用 LLM API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"LLM API 调用失败: {e}")
            return ""
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        扩展查询
        
        Args:
            query: 原始查询
        
        Returns:
            扩展后的查询对象
        """
        system_prompt = """你是一个查询扩展助手。接受简短的用户查询并将其扩展为更详细的版本：
1. 添加相关上下文和澄清
2. 包含相关术语和概念
3. 指定应涵盖的方面
4. 保持原始意图
5. 保持为单个连贯问题

将查询扩展为详细2-3倍，同时保持专注。"""
        
        expanded = self._call_llm(
            prompt=f"扩展此查询：{query}",
            system_prompt=system_prompt
        )
        
        if not expanded:
            expanded = query
        
        return ExpandedQuery(
            original=query,
            expanded=expanded,
            variations=[]
        )
    
    def generate_variations(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        生成查询变体
        
        Args:
            query: 原始查询
            num_variations: 变体数量
        
        Returns:
            查询变体列表
        """
        system_prompt = f"""生成此查询的{num_variations}种不同表述。
每个表述应该：
- 用不同的词汇表达相同的意图
- 关注不同的角度或方面
- 保持原意但使用不同措辞

仅返回{num_variations}个查询，每行一个，不要编号或其他格式。"""
        
        result = self._call_llm(
            prompt=f'查询："{query}"',
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        if result:
            # 清理结果
            variations = [
                line.strip()
                for line in result.split('\n')
                if line.strip() and not line.strip().startswith(('#', '-', '*', '1', '2', '3'))
            ]
            return variations[:num_variations]
        
        return []
    
    def expand_with_variations(
        self,
        query: str,
        num_variations: int = 3
    ) -> ExpandedQuery:
        """
        扩展查询并生成变体
        
        Args:
            query: 原始查询
            num_variations: 变体数量
        
        Returns:
            包含扩展和变体的查询对象
        """
        expanded = self.expand_query(query)
        expanded.variations = self.generate_variations(query, num_variations)
        
        return expanded
    
    def get_all_queries(self, expanded_query: ExpandedQuery) -> List[str]:
        """
        获取所有查询（原始 + 扩展 + 变体）
        
        Args:
            expanded_query: 扩展查询对象
        
        Returns:
            所有查询列表
        """
        queries = [expanded_query.original]
        
        if expanded_query.expanded != expanded_query.original:
            queries.append(expanded_query.expanded)
        
        queries.extend(expanded_query.variations)
        
        return list(set(queries))  # 去重


# 示例使用
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if api_key:
        expander = QueryExpander(api_key=api_key)
        
        query = "什么是 RAG？"
        result = expander.expand_with_variations(query)
        
        print(f"原始查询: {result.original}")
        print(f"\n扩展查询: {result.expanded}")
        print(f"\n变体:")
        for i, var in enumerate(result.variations, 1):
            print(f"  {i}. {var}")
        
        print(f"\n所有查询: {expander.get_all_queries(result)}")
    else:
        print("请设置 DASHSCOPE_API_KEY 环境变量")