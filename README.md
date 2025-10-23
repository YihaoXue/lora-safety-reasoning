# LORA IS ALL YOU NEED FOR SAFETY ALIGNMENT OF REASONING LLMS

> **Work in Progress**

This repository contains the data and code to reproduce the results from the paper [LoRA is All You Need for Safety Alignment of Reasoning LLMs](https://arxiv.org/abs/2507.17075)

Please check back later for updates.

## 📦 Installation

The minimal required packages are listed in `environment.yml`. You can run `conda env create -f environment.yml` for easy setup with Conda.

## ⚙️ Running Experiments

Each experiment consists of the following steps:
1. Perform safety alignment fine-tuning using either full-model fine-tuning or LoRA.  
2. Evaluate the safety of the fine-tuned models and the base model.  
3. Evaluate the reasoning performance of the fine-tuned models and the base model.

### 🎯 Safety Alignment Fine-tuning

Training is performed with `train.py`. All checkpoints and the final model will be saved in the `./finetuned_models` folder.

#### Full-model finetuning
##### Standard

Here is an example:

```bash
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
mode="full"
save_strategy="epoch"
per_device_bs=2
epochs=5

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --save_strategy $save_strategy \
    --per_device_bs $per_device_bs \
    --model_name $model_name \
    --epochs $epochs \
    --mode $mode
```

##### Training with DeepSpeed ZeRO-3

Set up the DeepSpeed configuration JSON file as needed, and pass it to the command via `--ds_config`. Include the `--shard` flag. Below is an example of fine-tuning a 32B model using the example config file `ds_config_zero3_32b.json`:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
mode="full"
epochs=1
per_device_bs=1
save_strategy="no"

CUDA_VISIBLE_DEVICES=0,1 deepspeed \
    train.py \
    --ds_config ds_config_zero3_32b.json \
    --model_name "$model_name" \
    --epochs "$epochs" \
    --mode "$mode" \
    --save_strategy $save_strategy \
    --per_device_bs $per_device_bs \
    --shard
```

#### LoRA Finetuning

We set the LoRA configuration through the `--mode` argument.  
Here are a few options:

- `lora_qkvo_mlp_r{int}` — Apply LoRA to both attention and MLP layers, with the specified rank *r*.  
- `lora_mlp_r{int}` — Apply LoRA only to MLP layers.  
- `lora_{string}_only_r{int}` — Apply LoRA only to a specific submodule within the MLP. The `{string}` can be one of `up_proj`, `down_proj`, or `gate_proj`.  
- `lora_{string}_only_from{int}_to{int}_r{int}` — Similar to the above, but restricts LoRA to specific layer indices.  
- `full` — Full-model fine-tuning instead of LoRA.  
- You can also find other variations in the definition of `parse_config_string()` in `train.py`, which includes several LoRA regularization methods that we explored.

Below is an example of applying LoRA only to the up-projection layers with layer indices from 16 to 31, with r=1.
```bash
model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
save_strategy="epoch"
per_device_bs=2
mode="lora_up_proj_only_from16_to31_r1"
epochs=10

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --per_device_bs $per_device_bs --model_name $model_name --mode $mode --epochs $epochs --save_strategy $save_strategy
```

### 🛡️ Safety Evaluation

#### 1. Sampling responses

The first step is to sample responses from the model and save them using `sample_responses.py`.

For a **LoRA model**, you need to provide both:
- the path to the saved PEFT LoRA weights via `--lora_path`, and  
- the base model path via `--model_path`.
The responses will be saved in `strongreject_responses.json` under the directroy of `--lora_path`. 

The responses will be saved in `strongreject_responses.json` under the directory specified by `--lora_path`.


For a **non-LoRA model** (e.g., a full-model fine-tuned model or the base model itself), you only need to specify `--model_path`. The responses will be saved in `strongreject_responses.json` under the directory specified by `--model_path`.

Below is an example of evaluating all checkpoints for a LoRA model that was trained for 10 epochs:

```bash
dataset_name="walledai/StrongREJECT"
size="14B"
model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-$size"
lora_name="lora_up_proj_only_from16_to31_r1_epochs_10"
batch_size=4

for ckpt_id in {500..5000..500}; do
  echo "Running for checkpoint $ckpt_id"
  lora_path="./finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-$size/$lora_name/checkpoint-$ckpt_id"
  CUDA_VISIBLE_DEVICES=0,1 python sample_responses.py \
      --lora_path $lora_path \
      --model_path $model_path \
      --dataset_name $dataset_name \
      --batch_size $batch_size
done
```

#### 2. Evaluating Responses

The second step is to use `evaluate_safety.py` to evaluate the sampled responses using a safety evaluator (here, `meta-llama/Llama-Guard-3-8B`).  
The evaluation results will be saved in `strongreject_responses_safety_eval.json` in the same folder as the response file.

Below is an example continuing from the previous step, evaluating the responses sampled for each checkpoint:

```bash
dataset_name="walledai/StrongREJECT"
size="14B"
model_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-$size"
lora_name="lora_up_proj_only_from16_to31_r1_epochs_10"
batch_size=4

for ckpt_id in {500..5000..500}; do
  echo "Running safety evaluation for checkpoint $ckpt_id"
  lora_path="./finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-$size/$lora_name/checkpoint-$ckpt_id"
  response_file="${lora_path}/strongreject_responses.json"
  CUDA_VISIBLE_DEVICES=0,1 python evaluate_safety.py \
      --response_file $response_file \
      --batch_size $batch_size
done
```

### 🧠 Reasoning Evaluation

#### 1. Math and science benchmarks

We adapted the evaluation code from [Small-Model-Learnability-Gap](https://github.com/Small-Model-Gap/Small-Model-Learnability-Gap), which builds upon [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
First, navigate to the `./lm-evaluation-harness` folder.  
Then, follow the three steps below:

1. **[Only needed for LoRA models]**  
   Use `lora_conversion.py` to merge the PEFT LoRA adapter weights with the base model, creating a standard merged model.  
   You can also use this script later to delete the merged model when it’s no longer needed.

2. **Sample responses with `lm_eval`**  
   Use the `lm_eval` CLI to sample and evaluate responses from your model —  
   either the merged model (for LoRA), a saved full-model fine-tuned checkpoint, or the base model itself.

3. **Evaluate the responses**  
   Since the simple rule-based evaluation in `lm_eval` can misjudge some cases, we perform a second-stage evaluation:  
   - For **GPQA**, use `mcq_metric_gpqa.py`, which applies a more comprehensive rule-based matching we defined that handles many edge cases.  
   - For **AIME**, use `math_metric_llm_eval_general.py`. This follows the method from [Small-Model-Learnability-Gap](https://github.com/Small-Model-Gap/Small-Model-Learnability-Gap), which leverages `Qwen2.5-32B-Instruct` to compare model responses with the ground-truth answers.

Below is an example for evaluating checkpoints of a LoRA fine-tuned model across random seeds on **GPQA** and **AIME**.  
You may want to modify `output_path` depending on how you prefer to organize the results.


```bash
gpus="0,1,2,3"
num_gpus=$(echo $gpus | awk -F',' '{print NF}')
model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
batch_size="auto"
max_model_tokens=32768
max_gen_tokens=32768
model_args="tensor_parallel_size=1,data_parallel_size=$num_gpus,gpu_memory_utilization=0.97,max_model_len=$max_model_tokens"

lora_name="lora_up_proj_only_from16_to31_r1_epochs_10" 
list_seed=(0 1 2 3 4 5 6 7)
tasks=("gpqa_diamond_better_prompt" "AIME")

for ckpt_id in {500..5000..500} do
    for seed in "${list_seed[@]}"; do
        lora_path="../finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B/$lora_name/checkpoint-$ckpt_id"
        merged_path="../finetuned_models/deepseek-ai_DeepSeek-R1-Distill-Qwen-14B/$lora_name/checkpoint-$ckpt_id"_merged
        CUDA_VISIBLE_DEVICES=$gpus python lora_conversion.py --base_model_path $model --lora_model_path $lora_path
        echo $merged_path
        for task in "${tasks[@]}"; do
            output_path="results/seed_$seed/$task/$lora_name/checkpoint-$ckpt_id"
            CUDA_VISIBLE_DEVICES=$gpus lm_eval --model vllm \
                --model_args pretrained="$merged_path",$model_args \
                --gen_kwargs do_sample=true,temperature=0.6,top_p=0.95,max_gen_toks=$max_gen_tokens,seed=$seed \
                --tasks "$task" \
                --batch_size "$batch_size" \
                --log_samples \
                --trust_remote_code \
                --output_path "$output_path" \
                --apply_chat_template \
                --seed $seed
            
            SANTIZED_MODEL_SAVE_LABEL=$(echo ${merged_path} | sed 's/\//__/g')
            echo ${SANTIZED_MODEL_SAVE_LABEL}
            if [ "$task" == "gpqa_diamond_better_prompt" ]; then
                echo "Running rule-based matching"
                python mcq_metric_gpqa.py --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} --task ${task}
            else
                CUDA_VISIBLE_DEVICES=$gpus python math_metric_llm_eval_general.py --tensor_parallel_size $num_gpus --directory_path ${output_path}/${SANTIZED_MODEL_SAVE_LABEL} --task ${task}
            fi
        done
    done
    python lora_conversion.py --base_model_path $model --lora_model_path $lora_path --delete
done
```

We note that we use the `gpqa_diamond_better_prompt` task — our modified version where we adjust the prompt slightly to better ensure that the model outputs answers in the desired format.

#### 2. Coding Benchmarks

We adapted [EvalPlus](https://github.com/evalplus/evalplus) for **HumanEval** and **MBPP**.

The original implementation included a *response prefix* designed for earlier models that did not explicitly support intermediate thinking process. This prefix — for example, *“Below is a Python script with a self-contained function that
efficiently solves the problem and passes corresponding tests:”* — was prepended to model outputs during generation. We found that this disadvantages models good at thinking — including the base model and LoRA-fine-tuned models — since the forced prefix disrupts their expected output format (which should always begin with a thinking process before generating the final code).  
As a result, these models may skip the reasoning process entirely, leading to unreasonably low performance. Therefore, we remove the response prefix in our evaluation to make it compatible with thinking models. We will add the code for this part to the repository soon. 


## 📄 Citation
If you find this work useful, please cite:

```bibtex
@article{xue2025lora,
  title={LoRA is All You Need for Safety Alignment of Reasoning LLMs},
  author={Xue, Yihao and Mirzasoleiman, Baharan},
  journal={arXiv preprint arXiv:2507.17075},
  year={2025}
}