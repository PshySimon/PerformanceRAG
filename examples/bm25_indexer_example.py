#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25索引器使用示例

本示例展示如何使用BM25IndexerComponent进行文档索引和搜索：
1. 加载测试数据文件
2. 创建BM25索引器
3. 索引文档
4. 执行搜索
5. 获取特定文档
6. 管理索引
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.components.index.bm25_indexer import BM25IndexerComponent
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")


def load_test_data():
    """加载测试数据文件"""
    test_data_dir = "/Users/caixiaomeng/Projects/Python/PerformanceRag/test_cases/test_data"
    documents = []
    
    # 加载混合语言数据
    with open(os.path.join(test_data_dir, "mixed_language_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "mixed_lang_doc",
            "content": content,
            "metadata": {
                "title": "Multi-language Performance Analysis Guide",
                "category": "Performance Analysis",
                "language": "mixed",
                "tags": ["performance", "analysis", "multilingual"]
            }
        })
    
    # 加载语义分割数据
    with open(os.path.join(test_data_dir, "semantic_splitter_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "ai_development_doc",
            "content": content,
            "metadata": {
                "title": "人工智能在软件开发中的应用",
                "category": "AI Development",
                "language": "chinese",
                "tags": ["AI", "machine learning", "software development"]
            }
        })
    
    # 加载文本分割数据
    with open(os.path.join(test_data_dir, "text_splitter_data.txt"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "perf_tool_doc",
            "content": content,
            "metadata": {
                "title": "Linux性能分析工具Perf简介",
                "category": "Linux Tools",
                "language": "chinese",
                "tags": ["linux", "perf", "performance", "tools"]
            }
        })
    
    # 加载递归分割数据
    with open(os.path.join(test_data_dir, "recursive_splitter_data.md"), "r", encoding="utf-8") as f:
        content = f.read()
        documents.append({
            "id": "perf_guide_doc",
            "content": content,
            "metadata": {
                "title": "Linux性能分析工具Perf完整指南",
                "category": "Linux Tools",
                "language": "chinese",
                "format": "markdown",
                "tags": ["linux", "perf", "guide", "tutorial"]
            }
        })
    
    return documents


def main():
    """主函数"""
    logger = get_logger(__name__)
    logger.info("开始BM25索引器示例")
    
    try:
        # 1. 创建BM25索引器
        logger.info("创建BM25索引器...")
        bm25_indexer = BM25IndexerComponent(
            name="test_bm25_indexer",
            config={
                "index_name": "test_documents",
                "k1": 1.5,
                "b": 0.75,
                "epsilon": 0.25,
                "storage_path": "./data/bm25_index",
                "auto_save": True,
                "enable_debug": True
            }
        )
        
        # 2. 初始化索引器
        logger.info("初始化索引器...")
        bm25_indexer.initialize()
        
        # 3. 创建索引
        logger.info("创建索引...")
        create_result = bm25_indexer.create_index()
        if create_result:
            logger.info("索引创建成功")
        else:
            logger.error("索引创建失败")
            return
        
        # 4. 加载测试数据
        logger.info("加载测试数据...")
        documents = load_test_data()
        logger.info(f"加载了 {len(documents)} 个文档")
        
        # 5. 索引文档
        logger.info("开始索引文档...")
        index_result = bm25_indexer.index_documents(documents)
        if index_result:
            logger.info(f"成功索引 {len(documents)} 个文档")
        else:
            logger.error("文档索引失败")
            return
        
        # 6. 执行搜索测试
        logger.info("\n=== 搜索测试 ===")
        
        # 测试查询
        queries = [
            "性能分析工具",
            "perf命令使用",
            "人工智能代码生成",
            "machine learning algorithms",
            "CPU cycles performance",
            "缓存未命中",
            "深度学习模型"
        ]
        
        for query in queries:
            logger.info(f"\n搜索查询: '{query}'")
            search_results = bm25_indexer.search(query, top_k=3)
            
            if search_results:
                for i, result in enumerate(search_results, 1):
                    logger.info(f"  结果 {i}: {result['metadata']['title']} (得分: {result['score']:.4f})")
            else:
                logger.info("  未找到相关结果")
        
        # 7. 获取特定文档
        logger.info("\n=== 文档获取测试 ===")
        doc = bm25_indexer.get_document("ai_development_doc")
        
        if doc:
            logger.info(f"获取文档: {doc['metadata']['title']}")
            logger.info(f"内容长度: {len(doc['content'])} 字符")
        else:
            logger.info("未找到指定文档")
        
        # 8. 多索引操作演示
        logger.info("\n=== 多索引操作演示 ===")
        
        # 创建另一个索引
        logger.info("创建第二个索引...")
        second_index_result = bm25_indexer.create_index("secondary_index")
        if second_index_result:
            logger.info("第二个索引创建成功")
            
            # 在第二个索引中索引部分文档
            subset_docs = documents[:2]
            index_result = bm25_indexer.index_documents(subset_docs, "secondary_index")
            if index_result:
                logger.info(f"在第二个索引中成功索引 {len(subset_docs)} 个文档")
                
                # 在第二个索引中搜索
                search_results = bm25_indexer.search("性能分析", top_k=2, index_name="secondary_index")
                logger.info(f"在第二个索引中找到 {len(search_results)} 个结果")
        
        # 9. 索引统计信息
        logger.info("\n=== 索引统计 ===")
        logger.info(f"索引文档数: {len(bm25_indexer.doc_metadata)}")
        if bm25_indexer.bm25:
            logger.info(f"BM25索引已创建，文档数量: {len(bm25_indexer.documents)}")
            # 计算平均文档长度
            if bm25_indexer.documents:
                avg_length = sum(len(doc) for doc in bm25_indexer.documents) / len(bm25_indexer.documents)
                logger.info(f"平均文档长度: {avg_length:.2f} 个词")
        
        # 注意：不再演示删除索引操作，因为该方法已被移除
        logger.info("\n注意: delete_index 方法已被移除，如需删除索引请手动删除索引文件")
        
        logger.info("\nBM25索引器示例完成!")
        
    except Exception as e:
        logger.error(f"示例执行失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()