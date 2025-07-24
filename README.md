# 性能调优RAG检索项目

## 项目简介

性能调优RAG检索项目是一个专门针对性能优化领域的检索增强生成(RAG)系统。项目采用模块化设计，支持多种检索方式，主要用于处理和检索性能调优相关的技术文档、最佳实践和解决方案。

## 核心特性

- 🔧 **模块化设计**: 基于组件的架构，易于扩展和维护
- 🚀 **多种检索**: BM25、Elasticsearch、向量检索等
- 📝 **智能分割**: 支持多种文本分割策略
- 🔄 **Pipeline流程**: 通过YAML配置灵活组建处理流程
- 📊 **详细日志**: 完整的调试和监控信息
- ⚡ **性能专用**: 专门针对性能调优领域的数据和场景优化

## 支持的组件类型

### 1. 文档加载器 (Loader)

#### FileLoaderComponent
负责从文件系统加载文档。

**参数配置:**
- `path` (str): 文档路径，默认"./data"
- `file_types` (list): 支持的文件类型，默认[".txt", ".md", ".pdf", ".html"]
- `encoding` (str): 文件编码，默认"utf-8"
- `debug` (bool): 调试模式，默认false

**示例配置:**
```yaml
document_loader:
  type: "loader"
  name: "file"
  config:
    path: "./test_cases/test_data"
    file_types: [".txt", ".md"]
    encoding: "utf-8"
    debug: true
```

#### WebLoaderComponent
从网页加载文档内容。

**参数配置:**
- `urls` (list): 要爬取的URL列表
- `max_depth` (int): 爬取深度，默认1
- `timeout` (int): 请求超时时间(秒)，默认30
- `headers` (dict): HTTP请求头

### 2. 文本分割器 (Splitter)

#### TextSplitterComponent
基础文本分割器，支持多种分割策略。

**参数配置:**
- `chunk_size` (int): 文本块大小，必需参数
- `chunk_overlap` (int): 重叠大小，默认0
- `split_method` (str): 分割方法，可选值: "character", "sentence", "paragraph"
- `include_metadata` (bool): 是否包含元数据，默认true

#### HierarchicalSplitterComponent
层次化分割器，支持多级分割。

**参数配置:**
- `chunk_sizes` (list): 多级块大小，如[512, 256]
- `chunk_overlap` (int): 重叠大小，默认50
- `include_prev_next_rel` (bool): 是否包含前后关系，默认true

**示例配置:**
```yaml
text_splitter:
  type: "splitter"
  name: "hierarchical"
  config:
    chunk_sizes: [512, 256]
    chunk_overlap: 50
    include_metadata: true
    include_prev_next_rel: true
```

### 3. 索引器 (Indexer)

#### BM25IndexerComponent
BM25算法索引器，适用于关键词检索。

**参数配置:**
- `index_name` (str): 索引名称，默认"default_index"
- `k1` (float): BM25参数k1，控制词频饱和度，默认1.2，范围[1.0-2.0]
- `b` (float): BM25参数b，控制文档长度归一化，默认0.75，范围[0.0-1.0]
- `epsilon` (float): BM25参数epsilon，默认0.25
- `storage_path` (str): 索引存储路径，默认"./data/bm25_index"
- `auto_save` (bool): 自动保存索引，默认true
- `batch_size` (int): 批处理大小，默认100

#### ElasticsearchIndexerComponent
Elasticsearch索引器，支持全文检索和向量检索。

**参数配置:**
- `host` (str): ES主机地址，默认"localhost"
- `port` (int): ES端口，默认9200
- `username` (str): 用户名
- `password` (str): 密码
- `use_ssl` (bool): 是否使用SSL，默认false
- `verify_certs` (bool): 是否验证证书，默认true
- `index_name` (str): 索引名称
- `batch_size` (int): 批处理大小，默认50

**示例配置:**
```yaml
bm25_indexer:
  type: "indexer"
  name: "bm25"
  config:
    index_name: "performance_docs"
    k1: 1.5
    b: 0.75
    storage_path: "./data/bm25_index"
    auto_save: true
    debug: true
```

### 4. 检索器 (Retriever)

#### BM25RetrieverComponent
BM25检索器，与BM25IndexerComponent配套使用。

**参数配置:**
- `index_name` (str): 索引名称，必须与索引器一致
- `k1` (float): BM25参数k1，建议与索引器一致
- `b` (float): BM25参数b，建议与索引器一致
- `top_k` (int): 返回结果数量，默认10
- `storage_path` (str): 索引存储路径
- `similarity_threshold` (float): 相似度阈值，默认0.0
- `auto_load` (bool): 自动加载索引，默认true

#### ElasticsearchRetrieverComponent
Elasticsearch检索器。

**参数配置:**
- `search_type` (str): 检索类型，可选"text"、"vector"、"hybrid"
- `embedding` (dict): 向量化配置(当search_type为vector时)
- 其他参数与ElasticsearchIndexerComponent相同

### 5. 重排序器 (Reranker)

#### EmbeddingRerankerComponent
基于向量相似度的重排序器。

**参数配置:**
- `top_k` (int): 重排后返回数量，默认10
- `score_threshold` (float): 分数阈值，默认0.0
- `embedding_model` (str): 向量模型名称
- `batch_size` (int): 批处理大小，默认32

#### LLMRerankerComponent
基于大语言模型的重排序器。

**参数配置:**
- `model_name` (str): LLM模型名称
- `temperature` (float): 生成温度，默认0.1
- `max_tokens` (int): 最大token数，默认512

### 6. 查询优化器 (Query)

#### QueryExpansionComponent
查询扩展组件，丰富查询内容。

**参数配置:**
- `expansion_method` (str): 扩展方法，可选"synonym"、"llm"、"embedding"
- `max_expansions` (int): 最大扩展数量，默认3
- `temperature` (float): LLM温度(当使用LLM扩展时)，默认0.7

#### QueryDecompositionComponent
查询分解组件，将复杂查询分解为子查询。

**参数配置:**
- `max_sub_queries` (int): 最大子查询数量，默认3
- `decomposition_strategy` (str): 分解策略，可选"sequential"、"parallel"

### 7. 生成器 (Generator)

#### LLMGeneratorComponent
基于大语言模型的答案生成器。

**参数配置:**
- `model_name` (str): 模型名称，默认"default"
- `temperature` (float): 生成温度，默认0.7，范围[0.0-2.0]
- `max_tokens` (int): 最大生成token数，默认1024
- `system_prompt` (str): 系统提示词
- `use_context` (bool): 是否使用检索上下文，默认true

**示例配置:**
```yaml
llm_generator:
  type: "generator"
  name: "llm"
  config:
    model_name: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 1024
    system_prompt: "你是一个专业的性能调优专家"
```

## 组件扩展指南

### 1. 创建新组件

所有组件都需要继承对应的基类：

```python
from rag.components.base import Component
from rag.components.retriever.base_retriever import BaseRetrieverComponent

class MyCustomRetriever(BaseRetrieverComponent):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        # 自定义参数
        self.custom_param = config.get("custom_param", "default_value")
    
    def _do_initialize(self):
        """初始化逻辑"""
        # 初始化代码
        pass
    
    def retrieve(self, query: str, top_k: Optional[int] = None, **kwargs) -> List[Dict[str, Any]]:
        """实现检索逻辑"""
        # 检索实现
        return results
    
    def _connect_to_index(self):
        """连接到索引"""
        # 连接逻辑
        pass
```

### 2. 注册组件

在组件模块的`__init__.py`中注册：

```python
from rag.pipeline.registry import ComponentRegistry
from .my_custom_retriever import MyCustomRetriever

# 注册组件
ComponentRegistry.register("retriever", "my_custom", MyCustomRetriever)
```

### 3. 组件类型说明

- **loader**: 数据加载器，负责从各种数据源加载文档
- **splitter**: 文本分割器，将长文档分割成小块
- **indexer**: 索引器，构建文档索引
- **retriever**: 检索器，从索引中检索相关文档
- **reranker**: 重排序器，对检索结果进行重新排序
- **query**: 查询优化器，优化用户查询
- **generator**: 生成器，基于检索结果生成答案

## Pipeline配置完整指南

### 1. 配置文件结构说明

Pipeline配置文件采用YAML格式，主要包含以下几个部分：

```yaml
# 配置文件基本结构
pipeline_name:  # Pipeline名称
  components:   # 组件定义部分
    component_id:
      type: "component_type"    # 组件类型
      name: "component_name"    # 组件实现名称
      config:                   # 组件具体配置
        param1: value1
        param2: value2
  
  flow:         # 流程定义部分
    component1: ["component2", "component3"]  # 定义组件间的连接关系
    component2: ["component4"]
    component3: ["component4"]
    component4: []  # 空数组表示终点组件
  
  entry_point: "component1"   # 可选：指定默认入口点
```

### 2. 创建简单的BM25检索Pipeline

创建文件 `config/simple_bm25.yaml`：

```yaml
# 简单BM25检索Pipeline
simple_bm25:
  components:
    # 文档加载器
    doc_loader:
      type: "loader"
      name: "file"
      config:
        path: "./test_cases/test_data"
        file_types: [".txt", ".md"]
        encoding: "utf-8"
        debug: true
    
    # 文本分割器
    splitter:
      type: "splitter"
      name: "text"
      config:
        chunk_size: 512
        chunk_overlap: 50
        split_method: "character"
        include_metadata: true
    
    # BM25索引器
    indexer:
      type: "indexer"
      name: "bm25"
      config:
        index_name: "simple_perf_docs"
        k1: 1.2
        b: 0.75
        storage_path: "./data/bm25_index"
        auto_save: true
    
    # BM25检索器
    retriever:
      type: "retriever"
      name: "bm25"
      config:
        index_name: "simple_perf_docs"
        top_k: 5
        storage_path: "./data/bm25_index"
        similarity_threshold: 0.1
  
  # 定义两个独立的流程
  flow:
    # 索引流程：加载 -> 分割 -> 索引
    doc_loader: ["splitter"]
    splitter: ["indexer"]
    indexer: []
    
    # 检索流程：独立的检索器
    retriever: []
  
  entry_point: "doc_loader"
```

### 3. 创建完整的RAG Pipeline

创建文件 `config/complete_rag.yaml`：

```yaml
# 完整的性能调优RAG Pipeline
complete_rag:
  components:
    # 文档加载
    document_loader:
      type: "loader"
      name: "file"
      config:
        path: "./data/performance_docs"
        file_types: [".txt", ".md", ".pdf"]
        encoding: "utf-8"
        debug: false
    
    # 层次化分割
    hierarchical_splitter:
      type: "splitter"
      name: "hierarchical"
      config:
        chunk_sizes: [1024, 512]
        chunk_overlap: 100
        include_metadata: true
        include_prev_next_rel: true
    
    # BM25索引
    bm25_indexer:
      type: "indexer"
      name: "bm25"
      config:
        index_name: "performance_knowledge_base"
        k1: 1.5
        b: 0.75
        storage_path: "./data/bm25_index"
        auto_save: true
    
    # 查询扩展
    query_expander:
      type: "query"
      name: "expansion"
      config:
        expansion_method: "llm"
        max_expansions: 3
        temperature: 0.7
    
    # BM25检索
    bm25_retriever:
      type: "retriever"
      name: "bm25"
      config:
        index_name: "performance_knowledge_base"
        top_k: 10
        storage_path: "./data/bm25_index"
        similarity_threshold: 0.1
    
    # 结果重排
    embedding_reranker:
      type: "reranker"
      name: "embedding"
      config:
        top_k: 5
        score_threshold: 0.2
        embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
        batch_size: 32
    
    # 答案生成
    answer_generator:
      type: "generator"
      name: "llm"
      config:
        model_name: "gpt-3.5-turbo"
        temperature: 0.7
        max_tokens: 1024
        system_prompt: |
          你是一个专业的性能调优专家。请基于提供的上下文信息，
          为用户的性能问题提供准确、实用的解决方案。
          如果上下文中没有相关信息，请明确说明。
  
  flow:
    # 索引流程
    document_loader: ["hierarchical_splitter"]
    hierarchical_splitter: ["bm25_indexer"]
    bm25_indexer: []
    
    # 查询流程
    query_expander: ["bm25_retriever"]
    bm25_retriever: ["embedding_reranker"]
    embedding_reranker: ["answer_generator"]
    answer_generator: []
  
  entry_point: "document_loader"
```

### 4. 创建混合检索Pipeline

创建文件 `config/hybrid_search.yaml`：

```yaml
# 混合检索Pipeline（BM25 + 向量检索）
hybrid_search:
  components:
    # 文档加载
    doc_loader:
      type: "loader"
      name: "file"
      config:
        path: "./data/performance_docs"
        file_types: [".txt", ".md"]
    
    # 文本分割
    text_splitter:
      type: "splitter"
      name: "text"
      config:
        chunk_size: 512
        chunk_overlap: 50
    
    # BM25索引器
    bm25_indexer:
      type: "indexer"
      name: "bm25"
      config:
        index_name: "hybrid_perf_docs"
        storage_path: "./data/bm25_index"
    
    # Elasticsearch向量索引器
    es_indexer:
      type: "indexer"
      name: "elasticsearch"
      config:
        index_name: "hybrid_perf_docs_vector"
        host: "localhost"
        port: 9200
        embedding:
          type: "openai"
          model: "text-embedding-ada-002"
    
    # BM25检索器
    bm25_retriever:
      type: "retriever"
      name: "bm25"
      config:
        index_name: "hybrid_perf_docs"
        top_k: 5
        storage_path: "./data/bm25_index"
    
    # 向量检索器
    vector_retriever:
      type: "retriever"
      name: "elasticsearch"
      config:
        index_name: "hybrid_perf_docs_vector"
        search_type: "vector"
        top_k: 5
        host: "localhost"
        port: 9200
    
    # 结果融合器（自定义组件）
    result_fusion:
      type: "fusion"
      name: "rrf"  # Reciprocal Rank Fusion
      config:
        weights: [0.6, 0.4]  # BM25权重0.6，向量检索权重0.4
        top_k: 8
  
  flow:
    # 索引流程 - 并行索引
    doc_loader: ["text_splitter"]
    text_splitter: ["bm25_indexer", "es_indexer"]
    bm25_indexer: []
    es_indexer: []
    
    # 检索流程 - 并行检索后融合
    bm25_retriever: ["result_fusion"]
    vector_retriever: ["result_fusion"]
    result_fusion: []
```

### 5. 使用配置文件创建Pipeline

#### 方法1：使用Factory快速创建

```python
from rag.pipeline.factory import create_pipeline

# 创建简单BM25 Pipeline
pipeline = create_pipeline("simple_bm25")

# 执行索引流程
index_result = pipeline.run({}, entry_point="doc_loader")
print(f"索引完成，处理了 {index_result.get('document_count', 0)} 个文档")

# 执行检索流程
query_result = pipeline.run({
    "query": "如何优化CPU性能？",
    "top_k": 5
}, entry_point="retriever")

print(f"检索到 {query_result.get('result_count', 0)} 个相关文档")
for i, doc in enumerate(query_result.get('results', [])):
    print(f"{i+1}. {doc['content'][:100]}... (分数: {doc['score']:.3f})")
```

#### 方法2：使用Builder分步构建

```python
from rag.pipeline.builder import PipelineBuilder

# 创建Builder
builder = PipelineBuilder.from_config("complete_rag")

# 分步构建（可选，用于调试）
builder.load_config()
print("配置加载完成")

builder.validate_config()
print("配置验证通过")

builder.create_components()
print(f"组件创建完成: {list(builder.components.keys())}")

# 构建完整Pipeline
pipeline = builder.build()
print("Pipeline构建完成")

# 执行完整RAG流程
result = pipeline.run({
    "query": "Java应用内存泄漏如何排查？"
}, entry_point="query_expander")

print(f"生成的答案: {result.get('answer', '')}")
```

### 6. 配置文件最佳实践

#### 6.1 环境相关配置

```yaml
# 开发环境配置
development_config:
  components:
    doc_loader:
      type: "loader"
      name: "file"
      config:
        path: "./test_cases/test_data"  # 测试数据
        debug: true                     # 开启调试
    
    bm25_indexer:
      type: "indexer"
      name: "bm25"
      config:
        storage_path: "./data/dev_index"  # 开发环境索引路径
        debug: true

# 生产环境配置
production_config:
  components:
    doc_loader:
      type: "loader"
      name: "file"
      config:
        path: "/data/production/docs"    # 生产数据路径
        debug: false                     # 关闭调试
    
    bm25_indexer:
      type: "indexer"
      name: "bm25"
      config:
        storage_path: "/data/production/index"  # 生产环境索引路径
        batch_size: 500                          # 更大的批处理
        debug: false
```

#### 6.2 参数调优建议

```yaml
# 针对性能调优场景的参数优化
performance_tuned:
  components:
    # 针对技术文档的分割策略
    tech_splitter:
      type: "splitter"
      name: "hierarchical"
      config:
        chunk_sizes: [800, 400]    # 适合技术文档的块大小
        chunk_overlap: 80           # 保持代码和配置的连续性
        include_metadata: true
    
    # 针对性能关键词优化的BM25参数
    perf_bm25:
      type: "indexer"
      name: "bm25"
      config:
        k1: 1.6                     # 提高关键词权重
        b: 0.8                      # 考虑文档长度差异
        epsilon: 0.3                # 适当的平滑参数
    
    # 性能专家提示词
    perf_generator:
      type: "generator"
      name: "llm"
      config:
        system_prompt: |
          你是一个资深的系统性能调优专家，具有丰富的实战经验。
          请基于提供的技术文档，为用户提供：
          1. 问题的根本原因分析
          2. 具体的解决步骤
          3. 相关的监控和验证方法
          4. 预防类似问题的建议
          
          回答要具体、可操作，避免空泛的理论。
```

#### 6.3 错误处理和调试

```yaml
# 调试配置示例
debug_pipeline:
  components:
    debug_loader:
      type: "loader"
      name: "file"
      config:
        path: "./test_cases/test_data"
        debug: true                    # 开启详细日志
        
    debug_splitter:
      type: "splitter"
      name: "text"
      config:
        chunk_size: 256
        debug: true                    # 输出分割详情
        
    debug_indexer:
      type: "indexer"
      name: "bm25"
      config:
        index_name: "debug_index"
        debug: true                    # 显示索引过程
        auto_save: true
        
    debug_retriever:
      type: "retriever"
      name: "bm25"
      config:
        index_name: "debug_index"
        debug: true                    # 显示检索详情
        top_k: 3
  
  flow:
    debug_loader: ["debug_splitter"]
    debug_splitter: ["debug_indexer"]
    debug_indexer: []
    debug_retriever: []
```

### 7. 常见配置问题和解决方案

**Q: 如何添加新的检索算法？**
A: 继承`BaseRetrieverComponent`，实现`retrieve`和`_connect_to_index`方法，然后在注册表中注册。

**Q: Pipeline执行失败怎么办？**
A: 检查配置文件格式、组件参数、文件路径等，启用debug模式查看详细日志。

**Q: 如何实现混合检索？**
A: 配置多个检索器组件，在flow中并行执行，然后使用融合组件合并结果。

**Q: 索引文件在哪里？**
A: 默认在`./data/bm25_index/`目录下，可通过`storage_path`参数自定义。

        