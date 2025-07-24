from enum import Enum


class SplitMethod(Enum):
    """文本分割方法枚举"""

    CHARACTER = "character"
    WORD = "word"
    SENTENCE = "sentence"

    @classmethod
    def from_str(cls, value: str):
        """从字符串创建枚举值"""
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"无效的分割方法: {value}，有效值为: {valid_values}")


class SplitterType(Enum):
    """分割器类型枚举"""

    TEXT = "text"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    HIERARCHICAL_DOC = "hierarchical_doc"
    SENTENCE = "sentence"

    @classmethod
    def from_str(cls, value: str):
        """从字符串创建枚举值"""
        try:
            return cls(value)
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(f"无效的分割器类型: {value}，有效值为: {valid_values}")
