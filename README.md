# PerformanceRAG 部署指南

一个基于Elasticsearch的高性能RAG（检索增强生成）系统，支持Small2Big检索策略、混合检索（BM25+向量）、中文分词和重排序。

## 系统架构

- **检索器**: Elasticsearch + BGE-M3 embedding
- **重排序**: BGE-reranker-v2-m3
- **生成器**: 支持OpenAI API兼容的LLM
- **分词**: 支持jieba中文分词
- **检索策略**: Small2Big层级检索

## 部署步骤

### 1. 安装依赖

首先克隆项目并安装Python依赖：

```bash
git clone <your-repo-url>
cd PerformanceRag
pip install -r requirements.txt
```

主要依赖包括：

- `llama-index==0.12.49` - RAG框架
- `transformers==4.47.0` - 模型推理
- `vllm==0.8.5` - 高性能模型服务
- `elasticsearch` - 搜索引擎客户端
- `streamlit==1.36.0` - Web界面
- `jieba==0.42.1` - 中文分词

### 2. 部署模型服务

#### 2.1 部署Embedding模型（BGE-M3）

下载BGE-M3模型到 `models/bge-m3` 目录，然后启动服务：

```bash
bash scripts/deploy_embedding_model.sh
```

该脚本会在端口8001启动embedding服务：

```bash
vllm serve models/bge-m3 \
  --task embed \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --gpu-memory-utilization 0.4
```

#### 2.2 部署Reranker模型（BGE-reranker-v2-m3）

下载BGE-reranker-v2-m3模型到 `models/bge-reranker-v2-m3` 目录，然后启动服务：

```bash
bash scripts/deploy_reranker_model.sh
```

该脚本会在端口8002启动reranker服务：

```bash
vllm serve models/bge-reranker-v2-m3 \
  --task score --port 8002 \
  --host 0.0.0.0 \
  --dtype auto \
  --gpu-memory-utilization 0.4
```

#### 2.3 部署LLM模型

配置LLM服务，支持OpenAI API兼容格式。在 `config/llm.yaml` 中配置：

```yaml
default: "openai"

clients:
  openai:
    type: "openai"
    model: "qwen-plus"  # 或其他模型
    api_key: "your-api-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    max_tokens: 8192
    timeout: 60
```

### 3. 部署Elasticsearch

#### 3.1 安装Elasticsearch 8.x

**macOS (使用Homebrew):**

```bash
brew install elasticsearch
```

**Docker方式:**

```bash
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=true" \
  -e "ELASTIC_PASSWORD=your-password" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### 3.2 安装jieba分词插件（可选）

为了支持中文分词，建议安装elasticsearch-jieba插件：

```bash
# 下载对应版本的jieba插件
wget https://github.com/sing1ee/elasticsearch-jieba-plugin/releases/download/v8.11.0/elasticsearch-jieba-plugin-8.11.0.zip

# 安装插件
bin/elasticsearch-plugin install file:///path/to/elasticsearch-jieba-plugin-8.11.0.zip

# 重启Elasticsearch
```

#### 3.3 配置Elasticsearch连接

在 `config/es_search_pipeline.yaml` 中配置ES连接信息：

```yaml
components:
  es_retriever:
    config:
      index_name: "vector_performance_docs"
      host: "localhost"
      port: 9200
      username: "elastic"
      password: "your-password"
      use_ssl: true
      verify_certs: false
```

### 4. 数据切分与入库

#### 4.1 准备数据

将要索引的文档放在 `data/` 目录下，支持 `.txt` 和 `.md` 格式。

#### 4.2 配置数据处理管道

编辑 `config/datasource_producer_consumer_pipeline.yaml`：

```yaml
datasource:
  components:
    document_loader:
      config:
        path: "/path/to/your/data"  # 修改为你的数据路径
        file_types: [".txt", ".md"]
    
    text_splitter:
      config:
        chunk_sizes: [1536, 512, 128]  # Small2Big层级切分
        chunk_overlap: 50
        include_metadata: true

producer:
  components:
    openai_embedding:
      config:
        model: "models/bge-m3"
        api_base: "http://localhost:8001/v1"  # embedding服务地址
```

#### 4.3 执行数据入库

运行异步数据处理脚本：

```bash
python examples/async_datasource_producer_consumer_example.py
```

该脚本会：

1. 加载文档并进行层级切分（1536/512/128 tokens）
2. 生成向量embeddings
3. 索引到Elasticsearch
4. 支持断点续传和批量处理

### 5. 启动Streamlit应用

#### 5.1 配置检查

确保所有服务正常运行：

- Elasticsearch: `http://localhost:9200`
- Embedding服务: `http://localhost:8001`
- Reranker服务: `http://localhost:8002`
- LLM服务: 根据配置

#### 5.2 启动Web界面

```bash
streamlit run streamlit_chat_app.py
```

应用将在 `http://localhost:8501` 启动，提供：

- 智能问答界面
- 检索类型选择（文本/向量/混合）
- Small2Big检索策略
- 实时检索结果展示
- 源文档追溯

## 功能特性

### Small2Big检索策略

- **Small Chunk**: 128 tokens，用于精确匹配
- **Medium Chunk**: 512 tokens，平衡精度和上下文
- **Big Chunk**: 1536 tokens，提供完整上下文
- 自动扩展策略，从小块检索扩展到大块返回

### 混合检索

- **BM25文本检索**: 基于关键词匹配
- **向量语义检索**: 基于BGE-M3 embeddings
- **RRF融合**: 倒数排名融合算法
- **中文分词**: 支持jieba分词优化

### 重排序

- BGE-reranker-v2-m3模型
- 提升检索结果相关性
- 支持批量重排序

## 配置文件说明

- `config/datasource_producer_consumer_pipeline.yaml`: 数据处理管道配置
- `config/es_search_pipeline.yaml`: 搜索管道配置
- `config/embeddings.yaml`: Embedding模型配置
- `config/llm.yaml`: LLM模型配置

## 故障排除

### 常见问题

1. **Elasticsearch连接失败**
   - 检查ES服务是否启动
   - 验证用户名密码
   - 确认SSL配置

2. **模型服务无法访问**
   - 检查vLLM服务状态
   - 验证端口是否被占用
   - 查看GPU内存使用情况

3. **中文分词效果不佳**
   - 确认jieba插件安装
   - 检查分析器配置
   - 验证索引mapping

### 性能优化

1. **Elasticsearch优化**
   ```yaml
   settings:
     number_of_shards: 1
     number_of_replicas: 0
     refresh_interval: "30s"
   ```

2. **批量处理优化**
   - 调整 `bulk_size` 参数
   - 增加 `batch_size` 设置
   - 使用异步处理

3. **GPU内存优化**
   - 调整 `gpu-memory-utilization` 参数
   - 使用模型量化
   - 分布式部署

## 开发和测试

### 运行测试

```bash
# ES搜索管道测试
python examples/es_search_pipeline_example.py

# 语义分割测试
python examples/semantic_splitter_example.py
```

### 调试工具

```bash
# 调试ES索引结构
python scripts/debug_es_index_structure.py

# 调试jieba分词
python scripts/debug_es_jieba.py

# 更新ES jieba设置
python scripts/update_es_jieba_settings.py
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题，请通过GitHub Issues联系。