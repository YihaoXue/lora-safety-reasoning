gpus="4,5,6,7"

model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
lora_name="lora_mlp_r4_epochs_10"

tasks=("gpqa_diamond_better_prompt" "AIME")

batch_size="auto"
max_model_tokens=32768
max_gen_tokens=32768
model_args="tensor_parallel_size=1,data_parallel_size=4,gpu_memory_utilization=0.9,max_model_len=$max_model_tokens"

# running with the merged model due to memory issue
for seed in 0; do
    for ckpt_id in 5000; do
        lora_path="/home/yihaoxue/pycharm/SafetyTradeoff/finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B/$lora_name/checkpoint-$ckpt_id"
        python lora_conversion.py --base_model_path $model --lora_model_path $lora_path
        python lora_conversion.py --base_model_path $model --lora_model_path $lora_path --delete
    done
done