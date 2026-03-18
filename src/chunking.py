"""
策略1: 上下文感知分块 (Context-Aware Chunking)

不是在固定字符数处分割文档，而是分析语义边界和文档结构。
确保相关内容保持在一起，避免在思路中间切断句子。
"""

from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """分块数据结构"""
    content: str
    metadata: dict
    chunk_id: int
    
    def __post_init__(self):
        """添加标题上下文"""
        if 'heading' not in self.metadata:
            self.metadata['heading'] = ""


class ContextAwareChunker:
    """上下文感知分块器"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def _detect_semantic_boundaries(self, text: str) -> List[int]:
        """检测语义边界（标题、段落分隔等）"""
        boundaries = [0]
        
        # 检测标题模式
        heading_patterns = [
            r'\n#{1,6}\s+.+\n',  # Markdown 标题
            r'\n\d+\.\s+.+\n',    # 数字编号
            r'\n[一二三四五六七八九十]+[、.．]\s*.+\n',  # 中文编号
            r'\n【.+\】\n',       # 中文方括号标题
        ]
        
        for pattern in heading_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        # 检测段落分隔（双换行）
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.start())
        
        # 检测句子结束
        for match in re.finditer(r'[。！？\.\!\?]\s*', text):
            boundaries.append(match.end())
        
        return sorted(set(boundaries))
    
    def _find_best_split_point(
        self,
        text: str,
        target_position: int,
        boundaries: List[int]
    ) -> int:
        """找到最佳分割点（最接近目标位置的语义边界）"""
        if not boundaries:
            return target_position
        
        # 找到最接近目标位置的边界
        best_boundary = boundaries[0]
        min_distance = abs(best_boundary - target_position)
        
        for boundary in boundaries:
            distance = abs(boundary - target_position)
            if distance < min_distance:
                min_distance = distance
                best_boundary = boundary
        
        # 如果最近的边界太远，使用目标位置
        if min_distance > self.chunk_size * 0.3:
            return target_position
        
        return best_boundary
    
    def _extract_heading(self, text: str, position: int) -> str:
        """提取当前位置附近的标题"""
        # 向前查找最近的标题
        before_text = text[:position]
        
        # 查找 Markdown 标题
        heading_match = re.search(r'#+\s+(.+?)(?:\n|$)', before_text[::-1])
        if heading_match:
            return heading_match.group(1)[::-1].strip()
        
        # 查找数字编号
        heading_match = re.search(r'\n(\d+\.\s+.+?)(?:\n|$)', before_text[::-1])
        if heading_match:
            return heading_match.group(1)[::-1].strip()
        
        return ""
    
    def chunk_text(
        self,
        text: str,
        document_title: str = "",
        metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """
        对文本进行上下文感知分块
        
        Args:
            text: 要分块的文本
            document_title: 文档标题
            metadata: 额外的元数据
        
        Returns:
            分块列表
        """
        if metadata is None:
            metadata = {}
        
        # 检测语义边界
        boundaries = self._detect_semantic_boundaries(text)
        
        chunks = []
        chunk_id = 0
        start = 0
        
        while start < len(text):
            # 计算目标结束位置
            target_end = start + self.chunk_size
            
            # 找到最佳分割点
            if target_end >= len(text):
                end = len(text)
            else:
                end = self._find_best_split_point(text, target_end, boundaries)
            
            # 提取分块内容
            chunk_text = text[start:end].strip()
            
            # 如果分块太小，尝试合并到下一个
            if len(chunk_text) < self.min_chunk_size and end < len(text):
                # 查找下一个分割点
                next_end = min(end + self.chunk_size, len(text))
                next_boundary = self._find_best_split_point(
                    text, next_end, boundaries
                )
                chunk_text = text[start:next_boundary].strip()
                end = next_boundary
            
            if chunk_text:
                # 提取标题上下文
                heading = self._extract_heading(text, start) or document_title
                
                # 创建分块
                chunk_metadata = {
                    **metadata,
                    "document_title": document_title,
                    "heading": heading,
                    "start_char": start,
                    "end_char": end,
                }
                
                chunks.append(Chunk(
                    content=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                ))
                chunk_id += 1
            
            # 移动到下一个起始位置（考虑重叠）
            if end >= len(text):
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = end
        
        return chunks
    
    def chunk_documents(
        self,
        documents: List[dict],
        use_markdown_chunks: bool = True
    ) -> List[Chunk]:
        """
        对多个文档进行分块
        
        Args:
            documents: 文档列表，每个文档包含 'content' 和 'title' 字段
            use_markdown_chunks: 是否使用预分割的 Markdown 块（如果存在）
        
        Returns:
            所有文档的分块列表
        """
        all_chunks = []
        chunk_id = 0
        
        for doc in documents:
            # 检查是否有预分割的 Markdown 块
            if use_markdown_chunks and 'markdown_chunks' in doc and doc['markdown_chunks']:
                # 使用 LangChain Markdown 分割器的结果
                for md_chunk in doc['markdown_chunks']:
                    chunk_metadata = {
                        "document_title": doc.get('title', ''),
                        "source": md_chunk.get('metadata', {}).get('source', ''),
                        "type": md_chunk.get('metadata', {}).get('type', 'pdf_markdown'),
                        **md_chunk.get('metadata', {})
                    }
                    
                    # 提取标题上下文
                    heading = ""
                    if 'header_1' in chunk_metadata:
                        heading = chunk_metadata['header_1']
                    if 'header_2' in chunk_metadata:
                        heading += f" > {chunk_metadata['header_2']}" if heading else chunk_metadata['header_2']
                    chunk_metadata['heading'] = heading or doc.get('title', '')
                    
                    all_chunks.append(Chunk(
                        content=md_chunk['content'],
                        metadata=chunk_metadata,
                        chunk_id=chunk_id
                    ))
                    chunk_id += 1
            else:
                # 使用传统的语义分块
                chunks = self.chunk_text(
                    text=doc.get('content', ''),
                    document_title=doc.get('title', ''),
                    metadata=doc.get('metadata', {})
                )
                # 重新编号
                for chunk in chunks:
                    chunk.chunk_id = chunk_id
                    all_chunks.append(chunk)
                    chunk_id += 1
        
        return all_chunks


# 示例使用
if __name__ == "__main__":
    chunker = ContextAwareChunker(chunk_size=500, chunk_overlap=50)
    
    sample_text = """
# RAG 系统概述

RAG（Retrieval-Augmented Generation）是一种结合检索和生成的技术。

## 主要优势

1. 知识可更新：无需重新训练模型
2. 减少幻觉：基于检索到的事实生成答案
3. 可解释性：可以追溯答案来源

## 技术架构

RAG 系统通常包含以下组件：
- 文档处理：分块、向量化
- 向量数据库：存储和检索
- 大语言模型：生成最终答案
"""
    
    chunks = chunker.chunk_text(sample_text, "RAG 技术介绍")
    
    for chunk in chunks:
        print(f"\n--- Chunk {chunk.chunk_id} ---")
        print(f"Heading: {chunk.metadata.get('heading', 'N/A')}")
        print(f"Content: {chunk.content[:100]}...")