# utils.py
import argparse
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tempfile
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA fine-tuning")
    return parser.parse_args()

def create_optimizer_and_scheduler(model, learning_rate=5e-5, weight_decay=1e-4, num_warmup_steps=500, num_training_steps=5000):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return optimizer, scheduler

# === Apply tokenizer chat template ===
def format_prompt(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        raise RuntimeError(
            "The tokenizer does not support `apply_chat_template`. "
            "Make sure you are using a chat model with a defined chat template, "
            "or switch to a non-chat-style prompt format manually."
        )


def load_prompts(dataset_name, split, field_name, num_samples=None):
    dataset = load_dataset(dataset_name, split=split)

    if field_name not in dataset.column_names:
        raise ValueError(f"Field '{field_name}' not found in dataset columns: {dataset.column_names}")

    if num_samples is None or num_samples > len(dataset):
        num_samples = len(dataset)

    return dataset.select(range(num_samples))[field_name]


def lora_merge(base_model_path, lora_model_path):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")

    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_model_path)

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    return model


def lora_merge_and_load_vllm(base_model_path, lora_model_path, tensor_parallel_size):
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(dir='./')

    try:
        # Load and merge model
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")
        model = PeftModel.from_pretrained(model, lora_model_path)
        model = model.merge_and_unload()

        # Save to temp directory
        model.save_pretrained(temp_dir)

        # Save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.save_pretrained(temp_dir)

        # Load with vLLM
        from vllm import LLM
        llm = LLM(model=temp_dir, dtype="auto", tensor_parallel_size=tensor_parallel_size)

    finally:
        # Delete the temporary model files
        shutil.rmtree(temp_dir)

    return llm