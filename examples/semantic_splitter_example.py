#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于embedding的语义分割示例

这个示例展示如何使用语义分割器对文档进行智能切分。
语义分割器会根据文本的语义相似度来决定分割点，
而不是简单地按照字符数或句子数进行分割。
"""

import sys
from datetime import datetime
from pathlib import Path

from rag.pipeline.factory import create_pipeline
from utils.logger import get_logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = get_logger(__name__)


def save_split_results(documents, output_dir="test_cases/tmp_data"):
    """
    保存分割结果到指定目录

    Args:
        documents: 分割后的文档列表
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存每个分割块为单独的文件
    saved_files = []
    for i, doc in enumerate(documents):
        # 文件名格式：chunk_序号_时间戳.txt
        filename = f"chunk_{i+1:03d}_{timestamp}.txt"
        file_path = output_path / filename

        # 写入文件内容（只保存纯文本）
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])

        saved_files.append(filename)

    logger.info(f"分割结果已保存到 {output_path}")
    logger.info(f"共保存 {len(documents)} 个txt文件")

    return output_path, saved_files


def run_semantic_splitting_example():
    """
    运行语义分割示例
    """
    logger.info("开始运行基于embedding的语义分割示例")

    try:
        # 2. 创建并配置Pipeline
        logger.info("创建语义分割Pipeline...")
        pipeline = create_pipeline("semantic_splitter_example")

        # 3. 执行文档加载和分割
        logger.info("开始执行文档分割...")
        result = pipeline.run(input_data={}, entry_point="document_loader")

        # 4. 分析分割结果
        if "documents" in result:
            documents = result["documents"]
            logger.info(f"分割完成！共生成 {len(documents)} 个文档块")

            # 显示分割结果统计
            print("\n=== 语义分割结果统计 ===")
            print("原始文档数量: 1")
            print(f"分割后块数量: {len(documents)}")

            # 分析每个块的大小
            chunk_sizes = [len(doc["content"]) for doc in documents]
            print(f"平均块大小: {sum(chunk_sizes) / len(chunk_sizes):.1f} 字符")
            print(f"最小块大小: {min(chunk_sizes)} 字符")
            print(f"最大块大小: {max(chunk_sizes)} 字符")

            # 显示前几个分割块的内容预览
            print("\n=== 分割块内容预览 ===")
            for i, doc in enumerate(documents[:3]):  # 只显示前3个块
                content_preview = (
                    doc["content"][:100] + "..."
                    if len(doc["content"]) > 100
                    else doc["content"]
                )
                print(f"\n块 {i+1} (长度: {len(doc['content'])} 字符):")
                print(f"内容: {content_preview}")
                if "metadata" in doc:
                    print(f"元数据: {doc['metadata']}")

            if len(documents) > 3:
                print(f"\n... 还有 {len(documents) - 3} 个块未显示")

            # 保存分割结果到文件
            print("\n=== 保存分割结果 ===")
            output_path, saved_files = save_split_results(documents)
            print(f"✅ 分割结果已保存到: {output_path}")
            print(
                f"📄 保存的文件: {', '.join(saved_files[:3])}{'...' if len(saved_files) > 3 else ''}"
            )

            return documents, output_path

        else:
            logger.error("分割结果中没有找到documents字段")
            print(f"结果内容: {result}")
            return None, None

    except Exception as e:
        logger.error(f"语义分割示例执行失败: {e}")
        print(f"错误详情: {e}")
        return None, None


def compare_splitting_methods():
    """
    比较不同分割方法的效果
    """
    logger.info("比较不同分割方法的效果")

    # 这里可以扩展，比较语义分割与其他分割方法的差异
    # 例如：文本分割、递归分割等
    pass


def main():
    """
    主函数
    """
    print("基于Embedding的语义分割示例")
    print("=" * 50)

    # 检查配置
    print("\n请确保已正确配置以下内容:")
    print("1. 在 config/semantic_splitter_example.yaml 中设置正确的API密钥")
    print("2. 确保embedding服务可用（OpenAI API或本地服务）")
    print("3. 将你的文档放在 ./test_cases/test_data/ 目录下")

    input("\n按回车键继续...")

    # 运行示例
    documents, output_path = run_semantic_splitting_example()

    if documents and output_path:
        print("\n✅ 语义分割示例执行成功！")
        print(f"📁 分割文件保存在: {output_path}")
        print(f"📄 共生成 {len(documents)} 个txt文件")
        print("\n💡 提示:")
        print("- 语义分割会根据文本的语义相似度智能分割")
        print("- 相似主题的句子会被保持在同一个块中")
        print("- 可以通过调整 similarity_threshold 来控制分割粒度")
        print("- 较高的阈值会产生更多、更小的块")
        print("- 较低的阈值会产生更少、更大的块")
        print("- 分割结果已保存到 test_cases/tmp_data 目录")
    else:
        print("\n❌ 语义分割示例执行失败，请检查配置和日志")


if __name__ == "__main__":
    main()
