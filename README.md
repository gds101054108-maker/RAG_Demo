# RAG 系统 - 组合1: 生产就绪堆栈

基于文章《2026年构建RAG系统的核心策略：从60%到94%准确率》实现的完整 RAG 系统。

## 🎯 组合策略

本系统采用**组合1: 生产就绪堆栈**，预期效果：
- ✅ **92% 准确率**
- ⚡ **1.2秒 平均延迟**
- 💰 **约 $0.003 每次查询**

### 包含的策略

| 策略 | 作用 | 解决的问题 |
|------|------|-----------|
| **上下文感知分块** | 按语义边界分割文档 | 固定分块切断上下文 |
| **重排序** | 两阶段检索，交叉编码器重新评分 | 向量相似度 ≠ 语义相关性 |
| **查询扩展** | LLM 将简短查询扩展为详细版本 | 用户查询模糊 |
| **智能体 RAG** | AI 自主选择检索策略 | 不同问题需要不同策略 |

## 📁 项目结构

```
RAG_demo/
├── .env                    # 配置文件
├── requirements.txt        # 依赖包
├── config.py              # 配置管理
├── main.py                # 主入口
├── src/
│   ├── __init__.py
│   ├── chunking.py        # 上下文感知分块
│   ├── reranking.py       # 重排序模块
│   ├── query_expansion.py # 查询扩展
│   ├── agent_rag.py       # 智能体 RAG
│   ├── vector_store.py    # 向量存储
│   ├── document_loader.py # 文档加载器
│   └── rag_pipeline.py    # RAG 主流程
├── data/
│   ├── documents/         # 存放待摄取的文档
│   └── chroma_db/         # ChromaDB 数据库
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的阿里云百炼 API Key：

```env
DASHSCOPE_API_KEY=your_api_key_here
```

### 3. 摄取文档

```bash
# 创建文档目录
mkdir -p data/documents

# 将你的文档放入 data/documents 目录

# 摄取文档
python main.py ingest --dir ./data/documents
```

### 4. 查询

```bash
# 单次查询
python main.py query "什么是 RAG 系统？"

# 显示来源文档
python main.py query "什么是 RAG 系统？" --show-sources

# 交互模式
python main.py interactive
```

## 📖 使用示例

### 交互模式

```bash
$ python main.py interactive

============================================================
RAG 系统交互模式
============================================================
输入问题进行查询，输入 'quit' 或 'exit' 退出
输入 'stats' 查看系统统计信息
============================================================

请输入问题: RAG 系统有什么优势？

============================================================
查询: RAG 系统有什么优势？
============================================================
[智能体] 选择策略: search_knowledge_base
[扩展] 原始: RAG 系统有什么优势？
[扩展] 扩展后: RAG（检索增强生成）系统的主要优势是什么？请详细说明其在知识更新、减少幻觉...
[检索] 获取 20 个候选
[重排序] 返回前 5 个结果

------------------------------------------------------------
RAG 系统的主要优势包括：
1. **知识可更新**：无需重新训练模型即可更新知识库
2. **减少幻觉**：基于检索到的事实生成答案，降低虚构内容
3. **可解释性**：可以追溯答案来源，便于验证
------------------------------------------------------------
来源: 5 个文档
耗时: 1.23秒
```

### 命令行参数

```bash
# 禁用查询扩展
python main.py query "问题" --no-expansion

# 禁用重排序
python main.py query "问题" --no-reranking

# 禁用智能体决策
python main.py query "问题" --no-agent
```

## ⚙️ 配置选项

在 `.env` 文件中可以配置以下选项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DASHSCOPE_API_KEY` | 阿里云百炼 API Key | - |
| `EMBEDDING_MODEL` | 嵌入模型 | text-embedding-v3 |
| `LLM_MODEL` | 大语言模型 | qwen-flash |
| `CHUNK_SIZE` | 分块大小 | 512 |
| `CHUNK_OVERLAP` | 分块重叠 | 50 |
| `TOP_K` | 返回文档数 | 5 |
| `RERANK_CANDIDATES` | 重排序候选数 | 20 |

## 🔧 核心模块说明

### 1. 上下文感知分块 (chunking.py)

```python
from src.chunking import ContextAwareChunker

chunker = ContextAwareChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk_text(text, document_title="文档标题")
```

### 2. 重排序 (reranking.py)

```python
from src.reranking import HybridReranker

reranker = HybridReranker()
results = reranker.rerank(query, candidates, top_k=5)
```

### 3. 查询扩展 (query_expansion.py)

```python
from src.query_expansion import QueryExpander

expander = QueryExpander(api_key=api_key)
expanded = expander.expand_with_variations("什么是RAG？")
```

### 4. 智能体 RAG (agent_rag.py)

```python
from src.agent_rag import AgentRAG

agent = AgentRAG(api_key=api_key)
decision = agent.analyze_query("查询问题")
result = agent.retrieve("查询问题")
```

## 📊 性能优化建议

1. **批量摄取**：大量文档时使用批量模式
2. **缓存扩展查询**：对常见查询缓存扩展结果
3. **调整 top_k**：根据需求调整返回文档数
4. **异步处理**：生产环境可使用异步 API 调用

## 🐛 常见问题

### Q: 安装依赖时报错？
A: 确保使用 Python 3.9+，尝试使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: API 调用失败？
A: 检查 `.env` 中的 API Key 是否正确，网络是否通畅。

### Q: 检索结果不准确？
A: 尝试调整 `TOP_K` 和 `RERANK_CANDIDATES` 参数，或检查文档分块质量。

## 📝 更新日志

- **v1.0.0** (2026-03-16): 初始版本，实现组合1策略

## 📄 许可证

MIT License