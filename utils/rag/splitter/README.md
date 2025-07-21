# Splitter 模块

Splitter模块用于将文档分割成更小的chunk，以便于后续的向量化和检索。

## 功能特性

- **多种分割策略**: 支持字符分割、单词分割、句子分割
- **递归分割**: 使用多种分隔符进行层次化分割
- **语义分割**: 基于语义相似度进行分割（需要embedding模型）
- **灵活配置**: 支持通过配置文件自定义分割参数
- **元数据保留**: 自动保留原文档的元数据信息

## 模块结构

```
utils/rag/splitter/
├── __init__.py              # 模块导出
├── base_splitter.py         # 基础抽象类
├── text_splitter.py         # 文本分割器
├── recursive_splitter.py    # 递归分割器
├── semantic_splitter.py     # 语义分割器
├── splitter_factory.py      # 工厂类
└── README.md               # 本文档
```

## 快速开始

### 1. 基本使用

```python
from utils.rag.splitter import create_splitter
from utils.rag.loader.file_loader import FileLoader

# 加载文档
loader = FileLoader(path="./data/docs")
documents = loader.load()

# 创建分割器
splitter = create_splitter({
    "type": "text",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "split_method": "char"
})

# 分割文档
chunks = splitter.split(documents)
print(f"生成了 {len(chunks)} 个chunks")
```

### 2. 使用配置文件

```python
from utils.config import config

# 从配置文件创建分割器
splitter = create_splitter(config.splitter.text)
chunks = splitter.split(documents)
```

## 分割器类型

### 1. TextSplitter (文本分割器)

基于固定chunk大小的文本分割器，支持三种分割方法：

- **char**: 按字符数分割
- **word**: 按单词数分割  
- **sentence**: 按句子分割

```python
# 按字符分割
splitter = create_splitter({
    "type": "text",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "split_method": "char"
})

# 按单词分割
splitter = create_splitter({
    "type": "text", 
    "chunk_size": 100,
    "split_method": "word"
})

# 按句子分割
splitter = create_splitter({
    "type": "text",
    "chunk_size": 500,
    "split_method": "sentence"
})
```

### 2. RecursiveSplitter (递归分割器)

使用多种分隔符进行层次化分割，从最粗粒度到最细粒度：

```python
splitter = create_splitter({
    "type": "recursive",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
})
```

### 3. SemanticSplitter (语义分割器)

基于语义相似度进行分割，保持语义完整性：

```python
splitter = create_splitter({
    "type": "semantic",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-ada-002",  # 可选
    "similarity_threshold": 0.8
})
```

## 配置选项

### TextSplitter 配置

```yaml
text:
  type: text
  chunk_size: 1000          # chunk大小
  chunk_overlap: 200        # 重叠大小
  split_method: char        # 分割方法: char, word, sentence
  separator: "\n"           # 分隔符
  keep_separator: true      # 是否保留分隔符
```

### RecursiveSplitter 配置

```yaml
recursive:
  type: recursive
  chunk_size: 1000
  chunk_overlap: 200
  separators: ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
  keep_separator: true
```

### SemanticSplitter 配置

```yaml
semantic:
  type: semantic
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "text-embedding-ada-002"  # 可选
  similarity_threshold: 0.8
  min_chunk_size: 100
```

## 输出格式

分割后的每个chunk都是一个Document对象，包含以下元数据：

- `chunk_id`: chunk在文档中的序号
- `total_chunks`: 文档被分割成的总chunk数
- `chunk_size`: 当前chunk的字符数
- `split_method`: 使用的分割方法
- 原文档的所有元数据

## 使用建议

1. **选择合适的chunk大小**: 
   - 太小：可能丢失上下文信息
   - 太大：可能影响检索精度

2. **设置合理的重叠**:
   - 重叠太小：可能丢失跨chunk的信息
   - 重叠太大：会增加存储和计算成本

3. **根据文档类型选择分割器**:
   - 结构化文档：使用RecursiveSplitter
   - 长文本：使用TextSplitter
   - 需要语义完整性：使用SemanticSplitter

## 测试

运行测试：

```bash
# 运行所有splitter测试
pytest test_cases/splitter/

# 运行特定测试
pytest test_cases/splitter/test_text_splitter.py
```

## 示例

查看 `examples/splitter_example.py` 了解完整的使用示例。 