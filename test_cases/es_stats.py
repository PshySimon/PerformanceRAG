from elasticsearch import Elasticsearch


def get_es_stats():
    """统计ES索引vector_performance_docs的数据量和文件数"""

    # ES连接配置（从配置文件获取）
    es_config = {
        "hosts": ["https://localhost:9200"],
        "basic_auth": ("elastic", "sPxLec=NGSFmUT_7+74R"),
        "verify_certs": False,
    }

    try:
        # 连接ES
        es = Elasticsearch(**es_config)
        index_name = "vector_performance_docs"

        print(f"🔍 正在统计索引 {index_name} 的数据...")

        # 1. 获取索引基本统计信息
        stats = es.indices.stats(index=index_name)
        total_docs = stats["indices"][index_name]["total"]["docs"]["count"]
        index_size = stats["indices"][index_name]["total"]["store"]["size_in_bytes"]

        print("📊 索引基本信息:")
        print(f"   总文档数: {total_docs:,}")
        print(f"   索引大小: {index_size / (1024*1024):.2f} MB")

        # 2. 统计唯一文件数（通过metadata.source字段聚合）
        agg_query = {
            "size": 0,
            "aggs": {
                "unique_files": {
                    "terms": {
                        "field": "metadata.source.keyword",
                        "size": 10000,  # 假设文件数不超过10000
                    }
                },
                "file_doc_count": {
                    "terms": {"field": "metadata.source.keyword", "size": 10000}
                },
            },
        }

        try:
            response = es.search(index=index_name, body=agg_query)

            unique_files = response["aggregations"]["unique_files"]["buckets"]
            file_count = len(unique_files)

            print("\n📁 文件统计信息:")
            print(f"   唯一文件数: {file_count}")

            # 显示每个文件的文档数量
            print("\n📋 各文件文档分布:")
            for bucket in unique_files[:10]:  # 显示前10个文件
                filename = bucket["key"]
                doc_count = bucket["doc_count"]
                print(f"   {filename}: {doc_count} 个文档")

            if len(unique_files) > 10:
                print(f"   ... 还有 {len(unique_files) - 10} 个文件")

        except Exception as e:
            print(f"⚠️  聚合查询失败，可能是字段映射问题: {e}")
            print("尝试使用简单查询...")

            # 备用方案：扫描所有文档来统计
            scan_query = {"_source": ["metadata.source"], "size": 1000}

            files = set()
            response = es.search(index=index_name, body=scan_query, scroll="2m")
            scroll_id = response["_scroll_id"]

            # 处理第一批结果
            for hit in response["hits"]["hits"]:
                if (
                    "metadata" in hit["_source"]
                    and "source" in hit["_source"]["metadata"]
                ):
                    files.add(hit["_source"]["metadata"]["source"])

            # 继续滚动获取剩余数据
            while len(response["hits"]["hits"]) > 0:
                response = es.scroll(scroll_id=scroll_id, scroll="2m")
                for hit in response["hits"]["hits"]:
                    if (
                        "metadata" in hit["_source"]
                        and "source" in hit["_source"]["metadata"]
                    ):
                        files.add(hit["_source"]["metadata"]["source"])

            print("\n📁 文件统计信息 (扫描方式):")
            print(f"   唯一文件数: {len(files)}")
            print(f"   文件列表: {list(files)[:10]}")

        # 3. 获取索引映射信息
        mapping = es.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"]["properties"]

        print("\n🗂️  索引字段信息:")
        for field, config in properties.items():
            field_type = config.get("type", "object")
            print(f"   {field}: {field_type}")

    except Exception as e:
        print(f"❌ 连接ES失败: {e}")
        print("请检查ES服务是否运行，以及连接配置是否正确")


if __name__ == "__main__":
    get_es_stats()
