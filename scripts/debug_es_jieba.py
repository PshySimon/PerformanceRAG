#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试ES jieba插件配置和索引字段
"""

import json
import os
import sys

import urllib3
from elasticsearch import Elasticsearch

from utils.logger import get_logger, setup_logging

# 禁用urllib3的SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

setup_logging(level="INFO")
logger = get_logger(__name__)

# ES连接配置
ES_CONFIG = {
    "hosts": ["https://localhost:9200"],
    "basic_auth": ("elastic", "sPxLec=NGSFmUT_7+74R"),
    "verify_certs": False,
    "timeout": 60,
}


def check_index_structure(es, index_name="vector_performance_docs"):
    """检查索引结构和字段映射"""
    logger.info(f"\n=== 检查索引 {index_name} 的结构 ===")

    try:
        # 1. 检查索引是否存在
        if not es.indices.exists(index=index_name):
            logger.error(f"索引 {index_name} 不存在！")
            return

        # 2. 获取索引映射
        mapping = es.indices.get_mapping(index=index_name)
        logger.info("索引映射结构:")
        print(json.dumps(mapping, indent=2, ensure_ascii=False))

        # 3. 获取索引设置
        settings = es.indices.get_settings(index=index_name)
        logger.info("\n索引设置:")
        print(json.dumps(settings, indent=2, ensure_ascii=False))

        # 4. 检查文档总数
        count_result = es.count(index=index_name)
        total_docs = count_result["count"]
        logger.info(f"\n索引总文档数: {total_docs}")

        # 5. 检查有向量的文档数
        vector_count = es.count(
            index=index_name, body={"query": {"exists": {"field": "content_vector"}}}
        )
        logger.info(f"有向量的文档数: {vector_count['count']}")

        # 6. 检查没有向量的文档数
        no_vector_count = es.count(
            index=index_name,
            body={
                "query": {"bool": {"must_not": {"exists": {"field": "content_vector"}}}}
            },
        )
        logger.info(f"没有向量的文档数: {no_vector_count['count']}")

        # 7. 获取样本文档查看字段结构
        sample_docs = es.search(
            index=index_name, body={"query": {"match_all": {}}, "size": 3}
        )

        logger.info("\n=== 样本文档字段结构 ===")
        for i, doc in enumerate(sample_docs["hits"]["hits"]):
            logger.info(f"\n文档 {i+1} (ID: {doc['_id']}):")
            logger.info(f"字段列表: {list(doc['_source'].keys())}")

            # 检查关键字段
            source = doc["_source"]
            if "content" in source:
                content_preview = (
                    source["content"][:100] + "..."
                    if len(source["content"]) > 100
                    else source["content"]
                )
                logger.info(f"content字段预览: {content_preview}")

            if "content_vector" in source:
                vector = source["content_vector"]
                if isinstance(vector, list):
                    logger.info(
                        f"content_vector字段: 列表，长度={len(vector)}, 前5个值={vector[:5]}"
                    )
                else:
                    logger.info(f"content_vector字段: {type(vector)}, 值={vector}")
            else:
                logger.warning("❌ 该文档缺少content_vector字段")

        # 8. 测试搜索功能
        logger.info("\n=== 测试搜索功能 ===")
        test_queries = ["什么是裸金属", "PCF", "网络配置"]

        for query in test_queries:
            logger.info(f"\n测试查询: '{query}'")

            # 测试文本搜索
            try:
                text_result = es.search(
                    index=index_name,
                    body={"query": {"match": {"content": query}}, "size": 5},
                )
                logger.info(f"文本搜索结果数: {text_result['hits']['total']['value']}")
            except Exception as e:
                logger.error(f"文本搜索失败: {e}")

            # 测试分析器
            try:
                analyze_result = es.indices.analyze(
                    index=index_name, body={"field": "content", "text": query}
                )
                tokens = [token["token"] for token in analyze_result["tokens"]]
                logger.info(f"分词结果: {tokens}")
            except Exception as e:
                logger.error(f"分析器测试失败: {e}")

    except Exception as e:
        logger.error(f"检查索引结构失败: {e}")
        import traceback

        traceback.print_exc()


def debug_es_jieba():
    """调试ES jieba插件配置"""
    try:
        es = Elasticsearch(**ES_CONFIG)
        logger.info("连接到Elasticsearch成功")

        # 1. 检查集群信息
        cluster_info = es.info()
        logger.info(f"ES版本: {cluster_info['version']['number']}")

        # 2. 检查已安装的插件
        try:
            plugins = es.cat.plugins(format="json")
            logger.info("已安装的插件:")
            for plugin in plugins:
                logger.info(f"  - {plugin['name']}: {plugin['component']}")
        except Exception as e:
            logger.error(f"获取插件列表失败: {e}")

        # 3. 检查索引结构（新增功能）
        check_index_structure(es)

        # 4. 创建测试索引来检查可用的分析器
        test_index = "test_jieba_debug"

        # 删除可能存在的测试索引
        if es.indices.exists(index=test_index):
            es.indices.delete(index=test_index)

        # 尝试不同的jieba配置
        jieba_configs = [
            {
                "name": "jieba_tokenizer",
                "config": {
                    "analysis": {"analyzer": {"test_analyzer": {"tokenizer": "jieba"}}}
                },
            },
            {
                "name": "jieba_max_word",
                "config": {
                    "analysis": {
                        "analyzer": {"test_analyzer": {"type": "jieba_max_word"}}
                    }
                },
            },
            {
                "name": "jieba_smart",
                "config": {
                    "analysis": {"analyzer": {"test_analyzer": {"type": "jieba_smart"}}}
                },
            },
            {
                "name": "ik_max_word",
                "config": {
                    "analysis": {"analyzer": {"test_analyzer": {"type": "ik_max_word"}}}
                },
            },
            {
                "name": "ik_smart",
                "config": {
                    "analysis": {"analyzer": {"test_analyzer": {"type": "ik_smart"}}}
                },
            },
        ]

        logger.info("\n=== 测试分析器配置 ===")
        for config in jieba_configs:
            try:
                logger.info(f"\n测试配置: {config['name']}")

                # 创建测试索引
                es.indices.create(
                    index=f"{test_index}_{config['name']}",
                    body={
                        "settings": config["config"],
                        "mappings": {
                            "properties": {
                                "content": {"type": "text", "analyzer": "test_analyzer"}
                            }
                        },
                    },
                )

                # 测试分析器
                test_text = "我爱北京天安门，什么是裸金属服务器"
                analyze_result = es.indices.analyze(
                    index=f"{test_index}_{config['name']}",
                    body={"analyzer": "test_analyzer", "text": test_text},
                )

                tokens = [token["token"] for token in analyze_result["tokens"]]
                logger.info(f"✅ {config['name']} 成功! 分词结果: {tokens}")

                # 删除测试索引
                es.indices.delete(index=f"{test_index}_{config['name']}")

            except Exception as e:
                logger.error(f"❌ {config['name']} 失败: {e}")

    except Exception as e:
        logger.error(f"调试失败: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_es_jieba()
