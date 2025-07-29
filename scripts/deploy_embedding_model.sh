vllm serve models/bge-m3 \
  --task embed \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --gpu-memory-utilization 0.4
