"""
RAG 系统主入口

组合1: 生产就绪堆栈
- 上下文感知分块 + 重排序 + 查询扩展 + 智能体 RAG
- 目标: 92% 准确率, 1.2秒平均延迟

使用方法:
    # 摄取文档
    python main.py ingest --dir ./documents
    
    # 查询
    python main.py query "什么是 RAG 系统？"
    
    # 交互模式
    python main.py interactive
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from config import settings
from src.rag_pipeline import RAGPipeline
from src.document_loader import DocumentLoader


def create_pipeline() -> RAGPipeline:
    """创建 RAG 流水线"""
    return RAGPipeline(
        api_key=settings.dashscope_api_key,
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        vector_db_path=settings.vector_db_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        rerank_candidates=settings.rerank_candidates
    )


def cmd_ingest(args):
    """摄取文档命令"""
    pipeline = create_pipeline()
    
    # 加载文档
    print(f"正在从 {args.dir} 加载文档...")
    documents = DocumentLoader.load_directory(args.dir)
    
    if not documents:
        print("未找到任何文档")
        return
    
    print(f"加载了 {len(documents)} 个文档")
    
    # 摄取到向量数据库
    num_chunks = pipeline.ingest_documents(documents)
    
    print(f"\n完成! 共摄取 {num_chunks} 个分块")
    
    # 显示统计信息
    stats = pipeline.get_stats()
    print(f"向量数据库: {stats['collection_name']}")
    print(f"文档总数: {stats['document_count']}")


def cmd_query(args):
    """查询命令"""
    pipeline = create_pipeline()
    
    # 执行查询
    response = pipeline.query(
        query=args.question,
        use_expansion=not args.no_expansion,
        use_reranking=not args.no_reranking,
        use_agent=not args.no_agent
    )
    
    # 显示结果
    print("\n" + "=" * 60)
    print("答案:")
    print("=" * 60)
    print(response.answer)
    
    if args.show_sources:
        print("\n" + "=" * 60)
        print("来源文档:")
        print("=" * 60)
        for i, source in enumerate(response.sources, 1):
            source_file = source['metadata'].get('source', '未知')
            doc_title = source['metadata'].get('document_title', '未知')
            heading = source['metadata'].get('heading', '')
            
            print(f"\n[{i}] 来源: {source_file}")
            print(f"    标题: {doc_title}")
            if heading:
                print(f"    章节: {heading}")
            print(f"    分数: {source['score']:.4f}")
            print(f"    内容: {source['content'][:100]}...")
    
    print(f"\n耗时: {response.metadata['elapsed_time']:.2f}秒")


def cmd_interactive(args):
    """交互模式"""
    pipeline = create_pipeline()
    
    print("\n" + "=" * 60)
    print("RAG 系统交互模式")
    print("=" * 60)
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
    print("输入 'stats' 查看系统统计信息")
    print("=" * 60 + "\n")
    
    while True:
        try:
            query = input("\n请输入问题: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break
            
            if query.lower() == 'stats':
                stats = pipeline.get_stats()
                print(f"\n文档总数: {stats['document_count']}")
                continue
            
            # 执行查询
            response = pipeline.query(query)
            
            print("\n" + "-" * 60)
            print(response.answer)
            print("-" * 60)
            print(f"来源: {response.metadata['num_sources']} 个文档")
            print(f"耗时: {response.metadata['elapsed_time']:.2f}秒")
            
        except KeyboardInterrupt:
            print("\n\n再见!")
            break
        except Exception as e:
            print(f"\n错误: {e}")


def cmd_clear(args):
    """清空向量数据库"""
    pipeline = create_pipeline()
    pipeline.vector_store.clear_collection()
    print("向量数据库已清空")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="RAG 系统 - 组合1: 生产就绪堆栈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 摄取文档目录
    
    python main.py ingest --dir ./documents
    
    # 执行查询
    python main.py query "什么是 RAG 系统？"
    
    # 交互模式
    python main.py interactive
    
    # 清空数据库
    python main.py clear
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ingest 命令
    ingest_parser = subparsers.add_parser('ingest', help='摄取文档到向量数据库')
    ingest_parser.add_argument('--dir', required=True, help='文档目录路径')
    ingest_parser.set_defaults(func=cmd_ingest)
    
    # query 命令
    query_parser = subparsers.add_parser('query', help='执行查询')
    query_parser.add_argument('question', help='查询问题')
    query_parser.add_argument('--show-sources', action='store_true', help='显示来源文档')
    query_parser.add_argument('--no-expansion', action='store_true', help='禁用查询扩展')
    query_parser.add_argument('--no-reranking', action='store_true', help='禁用重排序')
    query_parser.add_argument('--no-agent', action='store_true', help='禁用智能体决策')
    query_parser.set_defaults(func=cmd_query)
    
    # interactive 命令
    interactive_parser = subparsers.add_parser('interactive', help='交互模式')
    interactive_parser.set_defaults(func=cmd_interactive)
    
    # clear 命令
    clear_parser = subparsers.add_parser('clear', help='清空向量数据库')
    clear_parser.set_defaults(func=cmd_clear)
    
    args = parser.parse_args()
    
    # 加载环境变量
    load_dotenv()
    
    # 检查 API Key
    if not settings.dashscope_api_key:
        print("错误: 请在 .env 文件中设置 DASHSCOPE_API_KEY")
        sys.exit(1)
    
    # 执行命令
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()