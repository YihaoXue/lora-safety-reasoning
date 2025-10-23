# train_sft.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
import argparse
import os
from models import get_orthogonal_peft_model
import re


def parse_config_string(config_str):
    """
    Parses the configuration string and returns its category and extracted parameters.

    Returns:
        tuple:
            - category: one of ["lora_qkvo_mlp", "lora_mlp", "lora_mlp_orthogonal", "full"]
            - params: dict containing the relevant parameters

    Raises:
        ValueError: if the config_str is not in a valid format.
    """

    # Category: lora_qkvo_mlp_r{int}
    match = re.fullmatch(r"lora_qkvo_mlp_r(\d+)", config_str)
    if match:
        return "lora_qkvo_mlp", {"r": int(match.group(1))}

    # Category: lora_mlp_orthogonal_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }

    # Category: lora_mlp_orthogonal_norm_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_norm_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_norm", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }

    # Category: lora_mlp_orthogonal_norm_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_down_norm_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_down_norm", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }

    # Category: lora_mlp_orthogonal_uniform_norm_both_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_uniform_norm_both_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_uniform_norm_both", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }
    
    # Category: lora_mlp_orthogonal_norm_both_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_norm_both_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_norm_both", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }
    
    # Category: lora_mlp_orthogonal_down_uniform_norm_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_down_uniform_norm_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_down_uniform_norm", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }
    
    # Category: lora_mlp_orthogonal_down_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_down_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_down", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }

    # Category: lora_mlp_orthogonal_down_uniform_r{int}_beta{float}_k{int}
    match = re.fullmatch(r"lora_mlp_orthogonal_down_uniform_r(\d+)_beta([\d.]+)_k(\d+)", config_str)
    if match:
        return "lora_mlp_orthogonal_down_uniform", {
            "r": int(match.group(1)),
            "beta": float(match.group(2)),
            "k": int(match.group(3))
        }

    # Category: lora_mlp_r{int}
    match = re.fullmatch(r"lora_mlp_r(\d+)", config_str)
    if match:
        return "lora_mlp", {"r": int(match.group(1))}
    
    # Category: lora_{string}_only_r{int}
    match = re.fullmatch(r"lora_(.+?)_only_r(\d+)", config_str)
    if match:
        return "lora_module", {"module": match.group(1), "r": int(match.group(2))}

    # Category: lora_{string}_only_from{int}_to{int}_r{int}
    match = re.fullmatch(r"lora_(.+?)_only_from(\d+)_to(\d+)_r(\d+)", config_str)
    if match:
        return "lora_module_layer", {"module": match.group(1), "from": int(match.group(2)), "to": int(match.group(3)), "r": int(match.group(4))}

    # Category: full
    if config_str == "full":
        return "full", {}

    # Invalid category
    raise ValueError(f"Invalid configuration string: '{config_str}'")


def get_model_family(model_name: str) -> str:
    model_name = model_name.lower()
    if model_name.startswith("deepseek-ai/"):
        return "DeepSeek"
    elif model_name.startswith("qwen/"):
        return "Qwen"
    else:
        return "Unknown or Other"


def get_preprocess(model_family, tokenizer=None):

    if model_family == "DeepSeek":
        def preprocess(example):
            # Extract assistant's answer from the "refusal" field
            text = example["refusal"]
            if "Answer:" in text:
                response = text.split("Answer:", 1)[1]
            else:
                raise ValueError
            
            prompt = "<｜User｜>" + example["prompt"] + "<｜Assistant｜>"
            completion = "<think>" + "I should not answer this question!" + "</think>" + response + "<｜end▁of▁sentence｜>"

            return {
                "prompt": prompt,
                "completion": completion
            }
        
    elif model_family == "Qwen":
        def split_chat_response(text: str):
            """
            Split the chat-formatted string into:
            - prefix: everything before the assistant's response
            - response: the assistant's response and after
            """
            marker = "<|im_start|>assistant\n"
            start_idx = text.find(marker)
            if start_idx == -1:
                raise ValueError("No assistant start token found.")

            # Move index to just after the marker
            split_idx = start_idx + len(marker)
            prefix = text[:split_idx]
            response = text[split_idx:]
            return prefix, response
        
        def preprocess(example):
            text = example["refusal"]

            if "Answer:" in text:
                response = text.split("Answer:", 1)[1]
            else:
                raise ValueError

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": response}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt, completion = split_chat_response(text)

            return {
                "prompt": prompt,
                "completion": completion
            }
    else:
        raise NotImplementedError
    
    return preprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1) 
    parser.add_argument("--mode", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--save_strategy", type=str) 
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset_name", type=str, default="TianshengHuang/DirectRefusal")
    parser.add_argument("--per_device_bs", type=int)
    parser.add_argument('--shard', action='store_true')
    parser.add_argument("--ds_config", type=str, default=None) 
    return parser.parse_args()

def main():
    args = parse_args()
    
    training_mode, lora_params = parse_config_string(args.mode)
    print(training_mode)
    print(lora_params)

    output_dir = os.path.join(
        "finetuned_models",
        args.model_name.replace("/", "_"),
        f"{args.mode}_epochs_{args.epochs}"
    )

    if os.path.exists(output_dir):
        if len(os.listdir(output_dir)) != 0:
            raise RuntimeError(f"Output directory '{output_dir}' already exists and is not empty. Refusing to overwrite.")
        
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model_family = get_model_family(args.model_name)

    if model_family == "DeepSeek":
        tokenizer.pad_token = "<|fim_pad|>"  # Must not conflict with actual text tokens

    # Load your dataset
    raw_dataset = load_dataset("TianshengHuang/DirectRefusal", split="train")
    preprocess = get_preprocess(model_family, tokenizer)
    processed_dataset = raw_dataset.map(preprocess, remove_columns=raw_dataset.column_names)

    if args.shard:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        pretrained_model.gradient_checkpointing_enable()
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    # LoRA Config
    if training_mode == "lora_qkvo_mlp":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(pretrained_model, lora_config)
    elif training_mode == "lora_mlp":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(pretrained_model, lora_config)
    elif training_mode == "lora_module":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                lora_params["module"]
            ],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(pretrained_model, lora_config)
    elif training_mode == "lora_module_layer":

        if lora_params['module'] == "mlp":
            target_modules =[ 
                f"layers.{i}.mlp.gate_proj" for i in range(lora_params["from"], lora_params["to"]+1) 
            ] + [
                f"layers.{i}.mlp.up_proj" for i in range(lora_params["from"], lora_params["to"]+1)
            ] + [
                f"layers.{i}.mlp.down_proj" for i in range(lora_params["from"], lora_params["to"]+1)
            ]
        else:
            target_modules =[ 
                f"layers.{i}.mlp.{lora_params['module']}" for i in range(lora_params["from"], lora_params["to"]+1) 
            ]

        print("target modules:")
        print(target_modules)

        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= target_modules,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(pretrained_model, lora_config)
    elif training_mode == "lora_mlp_orthogonal":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_down":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, target_modules=["down_proj"], beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_down_uniform":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, weighted=False, target_modules=["down_proj"], beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_down_uniform_norm":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, normalize=True, weighted=False, target_modules=["down_proj"], beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_norm_both":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, normalize=True, weighted=True, orthogonal_mode="both", beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_uniform_norm_both":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, normalize=True, weighted=False, orthogonal_mode="both", beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_norm":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, normalize=True, beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "lora_mlp_orthogonal_down_norm":
        lora_config = LoraConfig(
            r=lora_params['r'],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules= [
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            task_type="CAUSAL_LM"
        )
        model = get_orthogonal_peft_model(pretrained_model, lora_config, normalize=True, target_modules=["down_proj"], beta=lora_params['beta'], k=lora_params['k'])
    elif training_mode == "full":
        model = pretrained_model

    # SFT Config
    if training_mode in ["lora_module", "lora_module_layer", "lora_mlp", "lora_qkvo_mlp", "lora_mlp_orthogonal", "lora_mlp_orthogonal_down", "lora_mlp_orthogonal_down_uniform", "lora_mlp_orthogonal_norm", "lora_mlp_orthogonal_down_norm", "lora_mlp_orthogonal_down_uniform_norm", "lora_mlp_orthogonal_uniform_norm_both", "lora_mlp_orthogonal_norm_both"]:
        if args.shard:
            raise NotImplementedError
        else:
            config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.per_device_bs,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_strategy=args.save_strategy,
                learning_rate=5e-5,
                weight_decay=1e-4,
                lr_scheduler_type="cosine",
                #warmup_steps=500,
                fp16=True,
                ddp_find_unused_parameters=False,
                report_to="none",
            )
    elif training_mode == "full":
        if args.shard:
            config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.per_device_bs,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_strategy=args.save_strategy,
                learning_rate=5e-5,
                weight_decay=1e-4,
                lr_scheduler_type="cosine",
                deepspeed=args.ds_config,
                report_to="none",
                fp16=True,
                bf16=False,
                save_only_model=True,
                gradient_checkpointing=True
            )
        else:
            config = SFTConfig(
                output_dir=output_dir,
                num_train_epochs=args.epochs, #5,
                per_device_train_batch_size=args.per_device_bs,
                gradient_accumulation_steps=1,
                logging_steps=10,
                save_strategy=args.save_strategy,
                learning_rate=5e-5,
                weight_decay=1e-4,
                lr_scheduler_type="cosine",
                #warmup_steps=500,
                fp16=False,
                bf16=True,
                ddp_find_unused_parameters=False,
                report_to="none",
            )
    else:
        raise NotImplementedError


    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_dataset,
        eval_dataset=processed_dataset,
        args=config
    )

    trainer.train()

    # === Clean up orthogonal regularization buffers ===
    if training_mode == "lora_mlp_orthogonal":
        for name, module in trainer.model.named_modules():
            if isinstance(module, torch.nn.Module):
                if hasattr(module, "init_span"):
                    print(f"Removing buffer from: {name}")
                    del module._buffers["init_span"]

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
