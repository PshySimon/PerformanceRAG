#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Markdown 文档加载示例 - 重点展示 Metadata
演示如何使用优化后的 FileLoader 按标题+段落切分 Markdown 文档
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class EnhancedFileLoader:
    """增强版文件加载器，支持 Markdown 结构化解析"""

    def __init__(self, enable_markdown_parsing: bool = True):
        """初始化加载器"""
        self.enable_markdown_parsing = enable_markdown_parsing

        if self.enable_markdown_parsing:
            self.markdown_parser = MarkdownNodeParser()

    def _extract_heading_context(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """提取节点的标题上下文，为每个节点添加对应的最小级别标题"""
        documents = []
        current_headings = {}  # 存储当前各级别的标题

        for node in nodes:
            metadata = node.metadata.copy()
            text = node.text.strip()

            # 如果是标题节点
            if "heading_level" in metadata:
                heading_level = metadata["heading_level"]
                # 更新当前标题层级
                current_headings[heading_level] = text
                # 清除更深层级的标题
                keys_to_remove = [
                    k for k in current_headings.keys() if k > heading_level
                ]
                for k in keys_to_remove:
                    del current_headings[k]

                # 为标题节点添加标题路径
                heading_path = []
                for level in sorted(current_headings.keys()):
                    heading_path.append(current_headings[level])
                metadata["heading_path"] = heading_path
                metadata["current_heading"] = text
                metadata["is_heading"] = True
                metadata["node_type"] = "heading"
            else:
                # 为正文节点添加对应的最小级别标题
                if current_headings:
                    # 获取最深层级的标题作为当前段落的标题
                    max_level = max(current_headings.keys())
                    metadata["current_heading"] = current_headings[max_level]
                    metadata["heading_level"] = max_level

                    # 添加完整的标题路径
                    heading_path = []
                    for level in sorted(current_headings.keys()):
                        heading_path.append(current_headings[level])
                    metadata["heading_path"] = heading_path
                else:
                    metadata["current_heading"] = None
                    metadata["heading_path"] = []

                metadata["is_heading"] = False
                metadata["node_type"] = "content"

            # 添加内容统计信息
            metadata["content_length"] = len(text)
            metadata["word_count"] = len(text.split())
            metadata["line_count"] = len(text.split("\n"))

            documents.append({"content": text, "metadata": metadata})

        return documents

    def load_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载 Markdown 文件并按标题+段落切分"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 创建 Document 对象
        doc = Document(
            text=content,
            metadata={
                "source": file_path,
                "file_type": ".md",
                "file_name": os.path.basename(file_path),
            },
        )

        # 使用 MarkdownNodeParser 解析
        nodes = self.markdown_parser.get_nodes_from_documents([doc])

        # 提取标题上下文并返回
        return self._extract_heading_context(nodes)


def print_metadata_focus(documents: List[Dict[str, Any]]):
    """重点打印每个节点的 Metadata 信息"""
    print(f"\n{'='*100}")
    print(f"🔍 METADATA 详细分析 - 总共 {len(documents)} 个节点")
    print(f"{'='*100}\n")

    for i, doc in enumerate(documents, 1):
        content = doc["content"]
        metadata = doc["metadata"]

        print(f"\n📋 节点 {i} - METADATA 详情")
        print(f"{'='*80}")

        # 🎯 核心 Metadata 信息
        print("\n🎯 【核心信息】")
        print(f"   • node_type: {metadata.get('node_type', 'unknown')}")
        print(f"   • is_heading: {metadata.get('is_heading', False)}")
        print(f"   • current_heading: {metadata.get('current_heading', 'None')}")

        if "heading_level" in metadata:
            print(f"   • heading_level: {metadata['heading_level']}")

        # 🗂️ 标题路径信息
        heading_path = metadata.get("heading_path", [])
        if heading_path:
            print("\n🗂️ 【标题路径】")
            for idx, path_item in enumerate(heading_path):
                indent = "   " + "  " * idx
                print(f"{indent}└─ H{idx+1}: {path_item}")

        # 📊 内容统计信息
        print("\n📊 【内容统计】")
        print(f"   • content_length: {metadata.get('content_length', 0)} 字符")
        print(f"   • word_count: {metadata.get('word_count', 0)} 词")
        print(f"   • line_count: {metadata.get('line_count', 0)} 行")

        # 📁 文件信息
        print("\n📁 【文件信息】")
        print(f"   • source: {metadata.get('source', 'Unknown')}")
        print(f"   • file_type: {metadata.get('file_type', 'Unknown')}")
        print(f"   • file_name: {metadata.get('file_name', 'Unknown')}")

        # 🔧 原始 llama_index metadata
        llama_metadata = {
            k: v
            for k, v in metadata.items()
            if k
            not in [
                "node_type",
                "is_heading",
                "current_heading",
                "heading_level",
                "heading_path",
                "content_length",
                "word_count",
                "line_count",
                "source",
                "file_type",
                "file_name",
            ]
        }
        if llama_metadata:
            print("\n🔧 【原始 LlamaIndex Metadata】")
            for key, value in llama_metadata.items():
                print(f"   • {key}: {value}")

        # 📄 内容预览
        content_preview = content.replace("\n", " ").strip()
        if len(content_preview) > 80:
            content_preview = content_preview[:80] + "..."
        print("\n📄 【内容预览】")
        print(f"   {content_preview}")

        # 📋 完整 Metadata JSON
        print("\n📋 【完整 Metadata JSON】")
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
        print(f"```json\n{metadata_json}\n```")

        print(f"\n{'-'*80}\n")


def analyze_metadata_patterns(documents: List[Dict[str, Any]]):
    """分析 Metadata 模式和统计"""
    print(f"\n{'='*100}")
    print("📈 METADATA 模式分析")
    print(f"{'='*100}")

    # 统计不同类型的节点
    headings = [doc for doc in documents if doc["metadata"].get("is_heading", False)]
    contents = [
        doc for doc in documents if not doc["metadata"].get("is_heading", False)
    ]

    print("\n📊 节点类型统计:")
    print(f"   • 总节点数: {len(documents)}")
    print(f"   • 标题节点: {len(headings)}")
    print(f"   • 内容节点: {len(contents)}")

    # 统计标题层级分布
    heading_levels = {}
    for doc in headings:
        level = doc["metadata"].get("heading_level", 0)
        heading_levels[level] = heading_levels.get(level, 0) + 1

    if heading_levels:
        print("\n🏷️ 标题层级分布:")
        for level in sorted(heading_levels.keys()):
            count = heading_levels[level]
            print(f"   • H{level}: {count} 个标题")

    # 分析内容长度分布
    content_lengths = [doc["metadata"].get("content_length", 0) for doc in documents]
    if content_lengths:
        print("\n📏 内容长度统计:")
        print(f"   • 最短: {min(content_lengths)} 字符")
        print(f"   • 最长: {max(content_lengths)} 字符")
        print(f"   • 平均: {sum(content_lengths) // len(content_lengths)} 字符")

    # 显示所有可用的 metadata 字段
    all_metadata_keys = set()
    for doc in documents:
        all_metadata_keys.update(doc["metadata"].keys())

    print("\n🔑 所有 Metadata 字段:")
    for key in sorted(all_metadata_keys):
        print(f"   • {key}")


def main():
    """主函数"""
    print("🔍 Markdown Metadata 详细分析器")
    print("重点展示每个节点的 Metadata 信息\n")

    # 指定要加载的文件
    file_path = "/Users/caixiaomeng/Projects/Python/PerformanceRag/test_cases/test_data/recursive_splitter_data.md"

    try:
        # 创建加载器
        loader = EnhancedFileLoader(enable_markdown_parsing=True)

        # 加载文档
        print(f"📂 正在加载文件: {file_path}")
        documents = loader.load_markdown_file(file_path)

        # 重点打印 Metadata 信息
        print_metadata_focus(documents)

        # 分析 Metadata 模式
        analyze_metadata_patterns(documents)

        print("\n✅ Metadata 分析完成！")
        print(
            "💡 关键信息: 每个节点都包含了丰富的 metadata，包括标题路径、层级、内容统计等"
        )

    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
