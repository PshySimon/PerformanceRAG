#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于左右互信息的新词发现算法

算法原理：
1. 左右互信息：衡量词汇边界的确定性
2. 内部凝聚度：衡量词汇内部的紧密程度
3. 词频统计：过滤低频噪声
4. 长度过滤：合理的词汇长度范围
"""

import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import jieba


class NewWordDiscovery:
    """新词发现器"""

    def __init__(
        self,
        min_freq: int = 5,
        min_len: int = 2,
        max_len: int = 6,
        min_pmi: float = 3.0,
        min_entropy: float = 1.0,
        top_k: int = 200,
    ):
        """
        初始化新词发现器

        Args:
            min_freq: 最小词频
            min_len: 最小词长
            max_len: 最大词长
            min_pmi: 最小互信息阈值
            min_entropy: 最小信息熵阈值
            top_k: 输出新词数量
        """
        self.min_freq = min_freq
        self.min_len = min_len
        self.max_len = max_len
        self.min_pmi = min_pmi
        self.min_entropy = min_entropy
        self.top_k = top_k

        # 统计数据
        self.word_count = Counter()  # 词频统计
        self.left_neighbors = defaultdict(Counter)  # 左邻字统计
        self.right_neighbors = defaultdict(Counter)  # 右邻字统计
        self.total_chars = 0  # 总字符数

        # 已知词汇（用于过滤）
        self.known_words = set()
        self._load_known_words()

        # 日志
        self.logger = logging.getLogger(__name__)

    def _load_known_words(self):
        """加载已知词汇（jieba词典）"""
        try:
            # 加载jieba默认词典
            jieba.initialize()
            # 这里可以添加更多已知词典的加载
            self.logger.info("已知词汇加载完成")
        except Exception as e:
            self.logger.warning(f"加载已知词汇失败: {e}")

    def read_corpus_from_directory(self, directory_path: str) -> str:
        """从目录读取所有txt文件内容"""
        corpus = []
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")

        txt_files = list(directory.glob("*.txt"))
        if not txt_files:
            raise ValueError(f"目录中没有找到txt文件: {directory_path}")

        self.logger.info(f"找到 {len(txt_files)} 个txt文件")

        for txt_file in txt_files:
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        corpus.append(content)
                        self.logger.debug(
                            f"读取文件: {txt_file.name}, 长度: {len(content)}"
                        )
            except Exception as e:
                self.logger.warning(f"读取文件失败 {txt_file}: {e}")

        full_corpus = "\n".join(corpus)
        self.logger.info(f"语料库总长度: {len(full_corpus)} 字符")
        return full_corpus

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 保留中文字符、数字、英文字母
        text = re.sub(r"[^\u4e00-\u9fff\w\s]", "", text)
        # 去除多余空白
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_candidates(self, text: str):
        """提取候选词并统计"""
        self.logger.info("开始提取候选词...")

        # 预处理文本
        text = self.preprocess_text(text)
        self.total_chars = len(text)

        # 滑动窗口提取n-gram
        for i in range(len(text)):
            for length in range(self.min_len, min(self.max_len + 1, len(text) - i + 1)):
                if i + length <= len(text):
                    candidate = text[i : i + length]

                    # 过滤条件
                    if not self._is_valid_candidate(candidate):
                        continue

                    # 统计词频
                    self.word_count[candidate] += 1

                    # 统计左邻字
                    if i > 0:
                        left_char = text[i - 1]
                        self.left_neighbors[candidate][left_char] += 1

                    # 统计右邻字
                    if i + length < len(text):
                        right_char = text[i + length]
                        self.right_neighbors[candidate][right_char] += 1

        self.logger.info(f"候选词提取完成，共 {len(self.word_count)} 个候选词")

    def _is_valid_candidate(self, candidate: str) -> bool:
        """判断候选词是否有效"""
        # 长度检查
        if len(candidate) < self.min_len or len(candidate) > self.max_len:
            return False

        # 不能全是数字或英文
        if candidate.isdigit() or candidate.isalpha():
            return False

        # 不能包含空格
        if " " in candidate:
            return False

        # 至少包含一个中文字符
        if not re.search(r"[\u4e00-\u9fff]", candidate):
            return False

        return True

    def calculate_pmi(self, word: str) -> Tuple[float, float]:
        """计算左右互信息"""
        if word not in self.word_count:
            return 0.0, 0.0

        word_freq = self.word_count[word]

        # 计算左互信息
        left_pmi = self._calculate_left_pmi(word, word_freq)

        # 计算右互信息
        right_pmi = self._calculate_right_pmi(word, word_freq)

        return left_pmi, right_pmi

    def _calculate_left_pmi(self, word: str, word_freq: int) -> float:
        """计算左互信息"""
        if len(word) < 2:
            return 0.0

        left_part = word[:-1]
        right_char = word[-1]

        left_freq = self.word_count.get(left_part, 0)
        right_freq = sum(
            self.word_count[w] for w in self.word_count if w.endswith(right_char)
        )

        if left_freq == 0 or right_freq == 0:
            return 0.0

        # PMI = log(P(xy) / (P(x) * P(y)))
        p_xy = word_freq / self.total_chars
        p_x = left_freq / self.total_chars
        p_y = right_freq / self.total_chars

        if p_x * p_y == 0:
            return 0.0

        return math.log2(p_xy / (p_x * p_y))

    def _calculate_right_pmi(self, word: str, word_freq: int) -> float:
        """计算右互信息"""
        if len(word) < 2:
            return 0.0

        left_char = word[0]
        right_part = word[1:]

        left_freq = sum(
            self.word_count[w] for w in self.word_count if w.startswith(left_char)
        )
        right_freq = self.word_count.get(right_part, 0)

        if left_freq == 0 or right_freq == 0:
            return 0.0

        # PMI = log(P(xy) / (P(x) * P(y)))
        p_xy = word_freq / self.total_chars
        p_x = left_freq / self.total_chars
        p_y = right_freq / self.total_chars

        if p_x * p_y == 0:
            return 0.0

        return math.log2(p_xy / (p_x * p_y))

    def calculate_entropy(self, word: str) -> Tuple[float, float]:
        """计算左右信息熵"""
        left_entropy = self._calculate_neighbor_entropy(self.left_neighbors[word])
        right_entropy = self._calculate_neighbor_entropy(self.right_neighbors[word])
        return left_entropy, right_entropy

    def _calculate_neighbor_entropy(self, neighbor_count: Counter) -> float:
        """计算邻字信息熵"""
        if not neighbor_count:
            return 0.0

        total = sum(neighbor_count.values())
        entropy = 0.0

        for count in neighbor_count.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def calculate_cohesion(self, word: str) -> float:
        """计算内部凝聚度（基于子串的互信息）"""
        if len(word) < 2:
            return 0.0

        min_pmi = float("inf")

        # 计算所有可能分割点的互信息
        for i in range(1, len(word)):
            left_part = word[:i]
            right_part = word[i:]

            left_freq = self.word_count.get(left_part, 0)
            right_freq = self.word_count.get(right_part, 0)
            word_freq = self.word_count.get(word, 0)

            if left_freq > 0 and right_freq > 0 and word_freq > 0:
                # 计算条件概率
                p_left_given_word = left_freq / word_freq if word_freq > 0 else 0
                p_right_given_word = right_freq / word_freq if word_freq > 0 else 0

                if p_left_given_word > 0 and p_right_given_word > 0:
                    pmi = math.log2(
                        word_freq / (left_freq * right_freq / self.total_chars)
                    )
                    min_pmi = min(min_pmi, pmi)

        return min_pmi if min_pmi != float("inf") else 0.0

    def filter_known_words(
        self, candidates: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """过滤已知词汇"""
        filtered = []
        for word, score in candidates:
            # 检查是否为jieba已知词汇
            jieba_words = list(jieba.cut(word, cut_all=False))
            if len(jieba_words) == 1 and jieba_words[0] == word:
                # 是jieba已知词汇，跳过
                continue

            filtered.append((word, score))

        return filtered

    def discover_new_words(self, directory_path: str) -> List[Tuple[str, Dict]]:
        """发现新词主函数"""
        self.logger.info("开始新词发现...")

        # 读取语料
        corpus = self.read_corpus_from_directory(directory_path)

        # 提取候选词
        self.extract_candidates(corpus)

        # 过滤低频词
        frequent_words = {
            word: freq
            for word, freq in self.word_count.items()
            if freq >= self.min_freq
        }

        self.logger.info(f"频次过滤后剩余 {len(frequent_words)} 个候选词")

        # 计算各项指标
        candidates = []
        for word in frequent_words:
            # 计算互信息
            left_pmi, right_pmi = self.calculate_pmi(word)
            min_pmi = min(left_pmi, right_pmi)

            # 计算信息熵
            left_entropy, right_entropy = self.calculate_entropy(word)
            min_entropy = min(left_entropy, right_entropy)

            # 计算凝聚度
            cohesion = self.calculate_cohesion(word)

            # 综合评分
            score = min_pmi * min_entropy * math.log(frequent_words[word])

            # 过滤条件
            if (
                min_pmi >= self.min_pmi
                and min_entropy >= self.min_entropy
                and cohesion > 0
            ):

                candidates.append(
                    (
                        word,
                        {
                            "freq": frequent_words[word],
                            "left_pmi": left_pmi,
                            "right_pmi": right_pmi,
                            "min_pmi": min_pmi,
                            "left_entropy": left_entropy,
                            "right_entropy": right_entropy,
                            "min_entropy": min_entropy,
                            "cohesion": cohesion,
                            "score": score,
                        },
                    )
                )

        self.logger.info(f"指标过滤后剩余 {len(candidates)} 个候选词")

        # 过滤已知词汇
        candidates_with_scores = [(word, stats["score"]) for word, stats in candidates]
        filtered_candidates = self.filter_known_words(candidates_with_scores)

        # 重新构建结果
        filtered_results = []
        filtered_words = {word for word, _ in filtered_candidates}
        for word, stats in candidates:
            if word in filtered_words:
                filtered_results.append((word, stats))

        self.logger.info(f"已知词过滤后剩余 {len(filtered_results)} 个候选词")

        # 按综合评分排序
        filtered_results.sort(key=lambda x: x[1]["score"], reverse=True)

        # 返回top_k结果
        return filtered_results[: self.top_k]

    def save_results(self, results: List[Tuple[str, Dict]], output_file: str):
        """保存结果到文件"""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("新词发现结果\n")
            f.write("=" * 50 + "\n")
            f.write(
                f"{'排名':<4} {'新词':<10} {'频次':<6} {'最小PMI':<8} {'最小熵':<8} {'凝聚度':<8} {'综合评分':<10}\n"
            )
            f.write("-" * 70 + "\n")

            for i, (word, stats) in enumerate(results, 1):
                f.write(
                    f"{i:<4} {word:<10} {stats['freq']:<6} "
                    f"{stats['min_pmi']:<8.3f} {stats['min_entropy']:<8.3f} "
                    f"{stats['cohesion']:<8.3f} {stats['score']:<10.3f}\n"
                )

        self.logger.info(f"结果已保存到: {output_file}")


def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description="新词发现工具")
    parser.add_argument("--input_dir", "-i", required=True, help="输入目录路径")
    parser.add_argument(
        "--output_file", "-o", default="new_words.txt", help="输出文件路径"
    )
    parser.add_argument("--min_freq", type=int, default=5, help="最小词频")
    parser.add_argument("--min_pmi", type=float, default=3.0, help="最小互信息")
    parser.add_argument("--min_entropy", type=float, default=1.0, help="最小信息熵")
    parser.add_argument("--top_k", type=int, default=200, help="输出新词数量")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 创建新词发现器
    discoverer = NewWordDiscovery(
        min_freq=args.min_freq,
        min_pmi=args.min_pmi,
        min_entropy=args.min_entropy,
        top_k=args.top_k,
    )

    try:
        # 发现新词
        results = discoverer.discover_new_words(args.input_dir)

        # 打印结果
        print(f"\n发现 {len(results)} 个新词:")
        print("=" * 50)
        for i, (word, stats) in enumerate(results[:20], 1):  # 只显示前20个
            print(
                f"{i:2d}. {word:<8} (频次:{stats['freq']}, PMI:{stats['min_pmi']:.2f}, "
                f"熵:{stats['min_entropy']:.2f}, 评分:{stats['score']:.2f})"
            )

        if len(results) > 20:
            print(f"... 还有 {len(results) - 20} 个新词")

        # 保存结果
        discoverer.save_results(results, args.output_file)

    except Exception as e:
        logging.error(f"新词发现失败: {e}")
        raise


if __name__ == "__main__":
    main()
