vllm serve models/bge-reranker-v2-m3 \
  --task score --port 8002 \
  --host 0.0.0.0 \
  --dtype auto \
  --gpu-memory-utilization 0.4