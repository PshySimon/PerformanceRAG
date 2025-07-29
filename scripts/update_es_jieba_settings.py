#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在线更新ES索引设置以支持jieba分词
无需重新索引数据的解决方案
"""

import os
import sys
from typing import Any, Dict

from elasticsearch import Elasticsearch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger(__name__)

# ES连接配置
ES_CONFIG = {
    "hosts": ["https://localhost:9200"],
    "basic_auth": ("elastic", "sPxLec=NGSFmUT_7+74R"),
    "verify_certs": False,
    "timeout": 60,
}

ORIGINAL_INDEX = "vector_performance_docs"
NEW_INDEX_NAME = "vector_performance_docs_jieba"
ALIAS_NAME = "vector_performance_docs_current"  # 使用不同的别名名


def create_jieba_settings() -> Dict[str, Any]:
    """创建支持IK分词的索引设置"""
    return {
        "analysis": {
            "analyzer": {
                "ik_analyzer": {
                    "type": "ik_max_word"
                },
                "ik_search_analyzer": {
                    "type": "ik_smart"
                }
            }
        },
        "number_of_shards": 1,
        "number_of_replicas": 0
    }

def create_jieba_mapping() -> Dict[str, Any]:
    """创建支持IK分词的字段映射"""
    return {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ik_analyzer",
                "search_analyzer": "ik_search_analyzer"
            },
            "content_jieba": {
                "type": "text",
                "analyzer": "ik_analyzer",
                "search_analyzer": "ik_search_analyzer"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "text",
                        "analyzer": "ik_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "file_path": {
                        "type": "text",
                        "analyzer": "ik_analyzer"
                    },
                    "filename": {
                        "type": "keyword",
                        "index": False      # filename不索引
                    },
                    "file_type": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks": {"type": "integer"},
                    "chunk_size": {"type": "integer"},
                    "split_method": {"type": "keyword"}
                }
            },
            "content_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine"
            },
            "timestamp": {"type": "date"}
        }
    }


def preprocess_with_jieba(text: str) -> str:
    """使用jieba预处理文本"""
    try:
        import jieba
        import jieba.analyse

        jieba.setLogLevel(20)  # 减少日志输出

        # 使用jieba进行分词
        tokens = list(jieba.cut(text.strip()))
        # 过滤空白、单字符和标点符号
        filtered_tokens = []
        for token in tokens:
            token = token.strip()
            if token and len(token) > 1 and not token.isspace():
                # 过滤纯标点符号
                if not all(ord(char) < 128 and not char.isalnum() for char in token):
                    filtered_tokens.append(token)

        return " ".join(filtered_tokens)
    except ImportError:
        logger.error("请安装jieba: pip install jieba")
        return text
    except Exception as e:
        logger.error(f"jieba分词失败: {e}")
        return text


# 删除这个函数，不需要了
# def preprocess_with_jieba(text: str) -> str:


# 在update_index_with_jieba()函数中，简化为直接reindex
def update_index_with_jieba():
    try:
        es = Elasticsearch(**ES_CONFIG)
        logger.info("连接到Elasticsearch成功")

        # 检查原索引是否存在
        if not es.indices.exists(index=ORIGINAL_INDEX):
            logger.error(f"原索引 {ORIGINAL_INDEX} 不存在")
            return False

        # 获取原索引统计信息
        stats = es.indices.stats(index=ORIGINAL_INDEX)
        total_docs = stats["indices"][ORIGINAL_INDEX]["total"]["docs"]["count"]
        logger.info(f"原索引文档数: {total_docs:,}")

        # 1. 创建新索引（支持jieba）
        logger.info(f"创建新索引 {NEW_INDEX_NAME}...")
        if es.indices.exists(index=NEW_INDEX_NAME):
            logger.info(f"删除已存在的索引 {NEW_INDEX_NAME}")
            es.indices.delete(index=NEW_INDEX_NAME)

        es.indices.create(
            index=NEW_INDEX_NAME,
            body={
                "settings": create_jieba_settings(),
                "mappings": create_jieba_mapping(),
            },
        )
        logger.info(f"新索引 {NEW_INDEX_NAME} 创建成功")

        # 2. 直接使用Reindex API复制数据，让ES jieba插件处理分词
        logger.info("开始复制数据...")
        reindex_body = {
            "source": {
                "index": ORIGINAL_INDEX
            },
            "dest": {
                "index": NEW_INDEX_NAME
            }
        }
        
        reindex_response = es.reindex(body=reindex_body, wait_for_completion=True)
        logger.info(f"数据复制完成，处理了 {reindex_response.get('total', 0)} 个文档")
        
        # 删除所有jieba预处理的代码，直接跳到别名创建
        # 3. 使用Update by Query为content字段添加jieba分词结果
        logger.info("开始为文档添加jieba分词结果...")

        # 分批处理文档
        batch_size = 100
        processed = 0

        # 滚动查询所有文档
        scroll_response = es.search(
            index=NEW_INDEX_NAME,
            scroll="2m",
            size=batch_size,
            body={"query": {"match_all": {}}},
        )

        scroll_id = scroll_response["_scroll_id"]
        hits = scroll_response["hits"]["hits"]

        while hits:
            # 准备批量更新操作
            bulk_body = []

            for hit in hits:
                doc_id = hit["_id"]
                content = hit["_source"].get("content", "")
                metadata = hit["_source"].get("metadata", {})

                # 使用jieba处理content
                content_jieba = preprocess_with_jieba(content)

                # 只处理file_path，不处理filename
                if "file_path" in metadata:
                    metadata["file_path_jieba"] = preprocess_with_jieba(
                        str(metadata["file_path"])
                    )

                # 添加更新操作
                bulk_body.append({"update": {"_index": NEW_INDEX_NAME, "_id": doc_id}})
                bulk_body.append(
                    {"doc": {"content_jieba": content_jieba, "metadata": metadata}}
                )

            # 执行批量更新
            if bulk_body:
                es.bulk(body=bulk_body)
                processed += len(hits)
                logger.info(f"已处理 {processed}/{total_docs} 个文档")

            # 获取下一批
            scroll_response = es.scroll(scroll_id=scroll_id, scroll="2m")
            hits = scroll_response["hits"]["hits"]

        # 清理scroll
        es.clear_scroll(scroll_id=scroll_id)

        # 4. 创建别名指向新索引
        logger.info("创建索引别名...")

        # 删除可能存在的旧别名
        if es.indices.exists_alias(name=ALIAS_NAME):
            aliases = es.indices.get_alias(name=ALIAS_NAME)
            old_indices = list(aliases.keys())
            for old_index in old_indices:
                es.indices.delete_alias(index=old_index, name=ALIAS_NAME)
            logger.info(f"删除旧别名 {ALIAS_NAME}")

        # 创建新别名
        es.indices.put_alias(index=NEW_INDEX_NAME, name=ALIAS_NAME)
        logger.info(f"创建别名 {ALIAS_NAME} 指向 {NEW_INDEX_NAME}")

        # 5. 验证结果
        logger.info("验证jieba分词效果...")
        test_query = "性能分析工具"

        # 测试jieba分词查询
        response = es.search(
            index=ALIAS_NAME,
            body={
                "query": {"match": {"content_jieba": test_query}},
                "size": 3,
                "highlight": {"fields": {"content_jieba": {}}},
            },
        )

        logger.info(
            f"测试查询 '{test_query}' 返回 {len(response['hits']['hits'])} 个结果"
        )
        for hit in response["hits"]["hits"][:2]:
            logger.info(f"文档ID: {hit['_id']}, 分数: {hit['_score']:.3f}")
            if "highlight" in hit:
                logger.info(f"高亮: {hit['highlight']}")

        logger.info("✅ jieba分词更新完成！")
        logger.info("\n📋 使用说明:")
        logger.info(f"  - 新索引名: {NEW_INDEX_NAME}")
        logger.info(f"  - 别名: {ALIAS_NAME}")
        logger.info(f"  - 原索引: {ORIGINAL_INDEX} (保持不变)")
        logger.info("\n🔍 查询字段:")
        logger.info("  - content_jieba: jieba分词的内容字段")
        logger.info("  - content: 原始内容字段")

        return True

    except Exception as e:
        logger.error(f"更新索引失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def update_config_file():
    """更新配置文件以使用新的索引"""
    try:
        config_file = "/Users/caixiaomeng/Projects/Python/PerformanceRag/config/es_search_pipeline.yaml"

        # 读取配置文件
        with open(config_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 替换索引名
        updated_content = content.replace(
            f'index_name: "{ORIGINAL_INDEX}"', f'index_name: "{ALIAS_NAME}"'
        )

        # 写回配置文件
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(updated_content)

        logger.info(f"✅ 配置文件已更新，现在使用别名 {ALIAS_NAME}")

    except Exception as e:
        logger.error(f"更新配置文件失败: {e}")


if __name__ == "__main__":
    logger.info("=== 开始更新ES索引以支持jieba分词 ===")

    # 检查jieba是否安装
    try:
        import jieba

        logger.info("✅ jieba已安装")
    except ImportError:
        logger.error("❌ 请先安装jieba: pip install jieba")
        sys.exit(1)

    # 执行更新
    success = update_index_with_jieba()

    if success:
        logger.info("\n🎉 索引更新成功！")

        # 更新配置文件
        update_config_file()

        logger.info("\n📝 下一步操作:")
        logger.info("1. 测试检索效果")
        logger.info("2. 确认无问题后可删除原索引")
        logger.info("3. 在代码中使用 content_jieba 字段进行jieba分词检索")

    else:
        logger.error("❌ 索引更新失败")
        sys.exit(1)
