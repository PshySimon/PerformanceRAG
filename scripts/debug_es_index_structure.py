#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试ES索引结构和向量字段
"""

import json
from elasticsearch import Elasticsearch
from utils.logger import get_logger, setup_logging

setup_logging(level="DEBUG")

# ES连接配置
ES_CONFIG = {
    "hosts": ["https://localhost:9200"],
    "basic_auth": ("elastic", "sPxLec=NGSFmUT_7+74R"),
    "verify_certs": False,
    "timeout": 30,
}

INDEX_NAME = "vector_performance_docs_jieba"

logger = get_logger(__name__)

def debug_index_structure():
    """调试索引结构"""
    try:
        es = Elasticsearch(**ES_CONFIG)
        logger.info("连接到Elasticsearch成功")
        
        # 1. 检查索引是否存在
        if not es.indices.exists(index=INDEX_NAME):
            logger.error(f"索引 {INDEX_NAME} 不存在")
            return False
            
        logger.info(f"✅ 索引 {INDEX_NAME} 存在")
        
        # 2. 获取索引mapping
        mapping = es.indices.get_mapping(index=INDEX_NAME)
        properties = mapping[INDEX_NAME]["mappings"]["properties"]
        
        logger.info("📋 索引字段结构:")
        for field_name, field_config in properties.items():
            field_type = field_config.get("type", "object")
            logger.info(f"   {field_name}: {field_type}")
            
            # 特别检查embedding字段
            if field_name == "embedding":
                logger.info(f"   🎯 找到embedding字段: {json.dumps(field_config, indent=4)}")
            elif "vector" in field_name.lower():
                logger.info(f"   🎯 找到向量相关字段 {field_name}: {json.dumps(field_config, indent=4)}")
        
        # 3. 检查是否有embedding字段
        has_embedding = "embedding" in properties
        has_vector_field = any("vector" in field.lower() for field in properties.keys())
        
        logger.info(f"\n🔍 向量字段检查结果:")
        logger.info(f"   embedding字段存在: {has_embedding}")
        logger.info(f"   其他向量字段存在: {has_vector_field}")
        
        if not has_embedding and not has_vector_field:
            logger.warning("❌ 未找到任何向量字段，这解释了为什么混合检索只有文本召回")
            logger.info("💡 建议: 需要重新索引数据并添加向量字段")
        
        # 4. 检查索引统计信息
        stats = es.indices.stats(index=INDEX_NAME)
        total_docs = stats["indices"][INDEX_NAME]["total"]["docs"]["count"]
        index_size = stats["indices"][INDEX_NAME]["total"]["store"]["size_in_bytes"]
        
        logger.info(f"\n📊 索引统计信息:")
        logger.info(f"   总文档数: {total_docs:,}")
        logger.info(f"   索引大小: {index_size / (1024*1024):.2f} MB")
        
        # 5. 抽样检查几个文档的字段
        logger.info(f"\n🔍 抽样检查文档字段:")
        sample_docs = es.search(
            index=INDEX_NAME,
            body={
                "query": {"match_all": {}},
                "size": 3,
                "_source": True
            }
        )
        
        for i, hit in enumerate(sample_docs["hits"]["hits"], 1):
            doc_fields = list(hit["_source"].keys())
            logger.info(f"   文档{i}字段: {doc_fields}")
            
            # 检查是否有embedding数据
            if "embedding" in hit["_source"]:
                embedding = hit["_source"]["embedding"]
                if isinstance(embedding, list) and len(embedding) > 0:
                    logger.info(f"     ✅ embedding字段有数据，维度: {len(embedding)}")
                else:
                    logger.info(f"     ❌ embedding字段为空")
            else:
                logger.info(f"     ❌ 文档中没有embedding字段")
        
        # 6. 测试向量检索查询
        logger.info(f"\n🧪 测试向量检索查询:")
        try:
            # 生成一个测试向量
            test_vector = [0.1] * 1024  # 假设是1024维向量
            
            vector_query = {
                "knn": {
                    "field": "embedding",
                    "query_vector": test_vector,
                    "k": 5,
                    "num_candidates": 10,
                },
                "size": 5,
            }
            
            vector_response = es.search(index=INDEX_NAME, body=vector_query)
            vector_hits = len(vector_response["hits"]["hits"])
            logger.info(f"   ✅ 向量检索测试成功，返回 {vector_hits} 个结果")
            
        except Exception as e:
            logger.error(f"   ❌ 向量检索测试失败: {e}")
            logger.info(f"   💡 这确认了索引缺少有效的embedding字段")
        
        return True
        
    except Exception as e:
        logger.error(f"调试索引结构失败: {e}")
        return False

def suggest_fix():
    """建议修复方案"""
    logger.info("\n🔧 修复建议:")
    logger.info("1. 如果索引缺少embedding字段，需要:")
    logger.info("   - 重新创建索引mapping，包含embedding字段")
    logger.info("   - 重新索引所有文档并生成向量")
    logger.info("2. 如果embedding字段存在但为空，需要:")
    logger.info("   - 运行向量化脚本为现有文档生成embedding")
    logger.info("3. 检查embedding服务是否正常工作")

if __name__ == "__main__":
    print("🔍 开始调试ES索引结构...")
    success = debug_index_structure()
    if success:
        suggest_fix()
    print("\n调试完成")