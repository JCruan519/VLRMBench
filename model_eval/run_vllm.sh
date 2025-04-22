MODEL="" # replace with your model path

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve $MODEL \
    --port 12453 \
    --dtype auto \
    --api-key token-abc123 \
    --served-model-name qwen2_5_vl_7b_vllm \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=4 \
