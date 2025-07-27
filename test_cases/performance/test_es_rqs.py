import random
import time
from datetime import datetime

import numpy as np
from elasticsearch import Elasticsearch, helpers

# ==== 配置区域 ====
ES_HOST = "https://localhost:9200"  # ES地址
ES_USER = "elastic"  # 用户名
ES_PASS = "sPxLec=NGSFmUT_7+74R"  # 密码

INDEX_NAME = "vector_perf_test"  # 测试索引名（保持不变）
VECTOR_DIM = 1024  # 向量维度
DOC_COUNT = 10000  # 总写入文档数
BULK_SIZE = 200  # 每批写入条数（匹配配置文件）

# ==== 初始化 ES 客户端 ====
es = Elasticsearch(
    ES_HOST, basic_auth=(ES_USER, ES_PASS), verify_certs=False, timeout=60
)

print(f"🔗 连接到 Elasticsearch: {ES_HOST}")
print(f"📊 准备测试索引: {INDEX_NAME}")

# ==== 删除旧索引（如存在） ====
if es.indices.exists(index=INDEX_NAME):
    print(f"🗑️  删除已存在的索引: {INDEX_NAME}")
    es.indices.delete(index=INDEX_NAME)

# ==== 创建索引（按照producer consumer配置） ====
print(f"🏗️  创建新索引: {INDEX_NAME}")
es.indices.create(
    index=INDEX_NAME,
    body={
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": 1024,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {"type": "object"},
                "timestamp": {"type": "date"},
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,  # 禁用副本提高性能
                "refresh_interval": "30s",  # 延长刷新间隔
            }
        },
    },
)


# ==== 生成随机文本内容 ====
def generate_random_content(length=200):
    """生成随机文本内容"""
    words = [
        "数据",
        "分析",
        "机器学习",
        "人工智能",
        "深度学习",
        "神经网络",
        "算法",
        "模型",
        "训练",
        "预测",
        "分类",
        "回归",
        "聚类",
        "优化",
        "特征",
        "向量",
        "矩阵",
        "概率",
        "统计",
        "数学",
        "计算",
        "处理",
    ]

    content_words = []
    for _ in range(length // 10):  # 大约生成指定长度的文本
        content_words.extend(random.choices(words, k=10))

    return " ".join(content_words[: length // 3])  # 控制文本长度


# ==== 构造测试文档 ====
def generate_doc(i):
    """生成测试文档，结构匹配producer consumer配置"""
    return {
        "_index": INDEX_NAME,
        "_source": {
            "content": generate_random_content(),
            "content_vector": np.random.rand(VECTOR_DIM).tolist(),
            "metadata": {
                "doc_id": f"test_doc_{i}",
                "source": "performance_test",
                "category": random.choice(["技术", "科学", "教育", "研究"]),
                "priority": random.randint(1, 5),
            },
            "timestamp": datetime.now().isoformat(),
        },
    }


# ==== 批量写入性能测试 ====
print("🚀 开始批量写入测试...")
print(f"📝 文档总数: {DOC_COUNT}, 批次大小: {BULK_SIZE}")

start_time = time.time()
success_count = 0
error_count = 0
batch_times = []

try:
    for i in range(0, DOC_COUNT, BULK_SIZE):
        batch_start = time.time()

        # 生成当前批次的文档
        chunk = [generate_doc(j) for j in range(i, min(i + BULK_SIZE, DOC_COUNT))]

        # 批量写入
        try:
            resp = helpers.bulk(es, chunk, request_timeout=60)
            success_count += resp[0]  # 成功插入的文档数
            if resp[1]:  # 如果有错误
                error_count += len(resp[1])
        except Exception as e:
            print(f"❌ 批次 {i//BULK_SIZE + 1} 写入失败: {e}")
            error_count += len(chunk)
            continue

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # 进度显示
        if (i // BULK_SIZE + 1) % 10 == 0:
            current_tps = len(chunk) / batch_time if batch_time > 0 else 0
            print(f"📊 已完成批次: {i//BULK_SIZE + 1}, 当前批次TPS: {current_tps:.2f}")

except KeyboardInterrupt:
    print("\n⚠️  测试被用户中断")

end_time = time.time()
total_elapsed = end_time - start_time

# ==== 手动刷新索引 ====
print("🔄 刷新索引...")
es.indices.refresh(index=INDEX_NAME)

# ==== 性能统计 ====
overall_tps = success_count / total_elapsed if total_elapsed > 0 else 0
avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
max_batch_time = max(batch_times) if batch_times else 0
min_batch_time = min(batch_times) if batch_times else 0

print("\n" + "=" * 60)
print("📈 性能测试结果")
print("=" * 60)
print(f"✅ 成功写入文档: {success_count:,} 条")
print(f"❌ 失败文档: {error_count:,} 条")
print(f"⏱️  总耗时: {total_elapsed:.2f} 秒")
print(f"🚀 整体TPS: {overall_tps:.2f} 条/秒")
print(f"📊 平均批次时间: {avg_batch_time:.3f} 秒")
print(f"⚡ 最快批次时间: {min_batch_time:.3f} 秒")
print(f"🐌 最慢批次时间: {max_batch_time:.3f} 秒")
print(f"📦 批次大小: {BULK_SIZE} 条")
print(f"🔢 总批次数: {len(batch_times)}")

# ==== 验证索引状态 ====
print("\n🔍 验证索引状态...")
try:
    index_stats = es.indices.stats(index=INDEX_NAME)
    doc_count = index_stats["indices"][INDEX_NAME]["total"]["docs"]["count"]
    index_size = index_stats["indices"][INDEX_NAME]["total"]["store"]["size_in_bytes"]
    print(f"📄 索引文档数: {doc_count:,}")
    print(f"💾 索引大小: {index_size / (1024*1024):.2f} MB")
except Exception as e:
    print(f"⚠️  获取索引统计失败: {e}")

# ==== 清理：删除测试索引 ====
print("\n🧹 清理测试数据...")
try:
    if es.indices.exists(index=INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
        print(f"✅ 已删除测试索引: {INDEX_NAME}")
    else:
        print(f"⚠️  索引 {INDEX_NAME} 不存在，无需删除")
except Exception as e:
    print(f"❌ 删除索引失败: {e}")

print("\n🎯 性能测试完成！")
