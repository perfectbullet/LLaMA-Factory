





### 利用 vLLM 部署 OpenAI API

```bash
API_PORT=8000 llamafactory-cli api examples/inference/llama3_vllm.yaml

# 启动特定模型推理
API_PORT=8010 llamafactory-cli api examples/inference/qwen-7b.yaml

# 启动原生
API_PORT=8010 llamafactory-cli api examples/inference/llama3-8b-chinese_vllm.yaml

# 启动微调后的, 并指定端口  8081
API_PORT=8010 llamafactory-cli api  examples/inference/llama3-8b-chinese_lora_sft.yaml

python src/api.py  examples/inference/qwen-7b.yaml
```
