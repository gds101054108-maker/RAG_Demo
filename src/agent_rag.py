"""
策略6: 智能体 RAG (Agentic RAG)

为 AI 智能体提供多个检索工具，让它根据查询自主选择使用哪个。
不同的问题需要不同的检索策略：
- 语义搜索：文档分块的语义相似性
- 完整文档：当分块缺乏上下文时
- 结构化数据：特定数据查询
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json


class RetrievalStrategy(Enum):
    """检索策略枚举"""
    SEMANTIC_SEARCH = "semantic_search"
    FULL_DOCUMENT = "full_document"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID_SEARCH = "hybrid_search"


@dataclass
class RetrievalTool:
    """检索工具定义"""
    name: str
    description: str
    strategy: RetrievalStrategy
    function: Callable


@dataclass
class AgentDecision:
    """智能体决策"""
    query: str
    selected_strategy: RetrievalStrategy
    reasoning: str
    tool_name: str


class AgentRAG:
    """智能体 RAG 系统"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "qwen-flash",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.tools: Dict[str, RetrievalTool] = {}
        self.vector_store = None
        
        # 注册默认工具
        self._register_default_tools()
    
    def _register_default_tools(self):
        """注册默认检索工具"""
        self.register_tool(RetrievalTool(
            name="search_knowledge_base",
            description="文档分块的语义搜索。适用于一般性问题，返回最相关的文档片段。",
            strategy=RetrievalStrategy.SEMANTIC_SEARCH,
            function=self._semantic_search
        ))
        
        self.register_tool(RetrievalTool(
            name="retrieve_full_document",
            description="检索完整文档。当分块缺乏上下文或需要完整信息时使用。",
            strategy=RetrievalStrategy.FULL_DOCUMENT,
            function=self._retrieve_full_document
        ))
        
        self.register_tool(RetrievalTool(
            name="keyword_search",
            description="关键词精确搜索。适用于查找特定术语、名称或代码。",
            strategy=RetrievalStrategy.KEYWORD_SEARCH,
            function=self._keyword_search
        ))
    
    def register_tool(self, tool: RetrievalTool):
        """注册检索工具"""
        self.tools[tool.name] = tool
    
    def set_vector_store(self, vector_store):
        """设置向量存储"""
        self.vector_store = vector_store
    
    def _semantic_search(self, query: str, limit: int = 5) -> str:
        """语义搜索实现"""
        if self.vector_store is None:
            return "错误：向量存储未初始化"
        
        results = self.vector_store.similarity_search(query, k=limit)
        return self._format_results(results)
    
    def _retrieve_full_document(self, document_title: str) -> str:
        """检索完整文档实现"""
        if self.vector_store is None:
            return "错误：向量存储未初始化"
        
        # 根据标题查找完整文档
        results = self.vector_store.search_by_metadata(
            {"document_title": document_title},
            limit=100
        )
        
        if not results:
            return f"未找到标题为 '{document_title}' 的文档"
        
        # 合并所有分块
        full_content = "\n\n".join([r['content'] for r in results])
        return f"**完整文档: {document_title}**\n\n{full_content}"
    
    def _keyword_search(self, keyword: str, limit: int = 5) -> str:
        """关键词搜索实现"""
        if self.vector_store is None:
            return "错误：向量存储未初始化"
        
        results = self.vector_store.keyword_search(keyword, k=limit)
        return self._format_results(results)
    
    def _format_results(self, results: List) -> str:
        """格式化搜索结果"""
        if not results:
            return "未找到相关内容"
        
        formatted = []
        for i, result in enumerate(results, 1):
            if hasattr(result, 'page_content'):
                content = result.page_content
                metadata = result.metadata
            else:
                content = result.get('content', '')
                metadata = result.get('metadata', {})
            
            formatted.append(f"[{i}] {content[:200]}...")
        
        return "\n\n".join(formatted)
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """调用 LLM API"""
        import requests
        
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
            "temperature": 0.3,
            "max_tokens": 1000
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
            return f"错误: {str(e)}"
    
    def analyze_query(self, query: str) -> AgentDecision:
        """分析查询并决定使用哪个工具"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        system_prompt = f"""你是一个 RAG 系统的智能检索协调器。
根据用户的查询，选择最合适的检索工具。

可用工具:
{tools_desc}

返回 JSON 格式:
{{
    "selected_tool": "工具名称",
    "reasoning": "选择该工具的原因"
}}"""
        
        result = self._call_llm(
            prompt=f"用户查询: {query}",
            system_prompt=system_prompt
        )
        
        try:
            # 尝试解析 JSON
            decision = json.loads(result)
            tool_name = decision.get("selected_tool", "search_knowledge_base")
            
            return AgentDecision(
                query=query,
                selected_strategy=self.tools.get(tool_name, list(self.tools.values())[0]).strategy,
                reasoning=decision.get("reasoning", ""),
                tool_name=tool_name
            )
        except:
            # 默认使用语义搜索
            return AgentDecision(
                query=query,
                selected_strategy=RetrievalStrategy.SEMANTIC_SEARCH,
                reasoning="默认选择语义搜索",
                tool_name="search_knowledge_base"
            )
    
    def retrieve(self, query: str, limit: int = 5) -> str:
        """
        智能检索
        
        Args:
            query: 用户查询
            limit: 返回结果数量
        
        Returns:
            检索结果
        """
        # 分析查询
        decision = self.analyze_query(query)
        
        print(f"[智能体决策] 使用工具: {decision.tool_name}")
        print(f"[推理] {decision.reasoning}")
        
        # 执行检索
        tool = self.tools.get(decision.tool_name)
        if tool:
            return tool.function(query, limit)
        
        return "错误：未找到合适的检索工具"
    
    def query_with_context(
        self,
        query: str,
        context: Optional[str] = None
    ) -> str:
        """
        带上下文的查询
        
        Args:
            query: 用户查询
            context: 额外上下文
        
        Returns:
            生成的答案
        """
        # 检索相关内容
        retrieved = self.retrieve(query)
        
        # 构建提示
        system_prompt = """你是一个专业的知识助手。
基于检索到的信息回答用户问题。
如果检索到的信息不足以回答问题，请诚实说明。"""
        
        full_query = f"""参考信息:
{retrieved}

{"额外上下文: " + context if context else ""}

用户问题: {query}

请基于参考信息给出详细、准确的回答。"""
        
        return self._call_llm(full_query, system_prompt)


# 示例使用
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    
    if api_key:
        agent = AgentRAG(api_key=api_key)
        
        # 测试查询分析
        test_queries = [
            "什么是 RAG 系统？",
            "请详细介绍文档《系统架构设计》的完整内容",
            "如何配置 API_KEY 参数？"
        ]
        
        for query in test_queries:
            print(f"\n查询: {query}")
            decision = agent.analyze_query(query)
            print(f"选择工具: {decision.tool_name}")
            print(f"推理: {decision.reasoning}")
    else:
        print("请设置 DASHSCOPE_API_KEY 环境变量")