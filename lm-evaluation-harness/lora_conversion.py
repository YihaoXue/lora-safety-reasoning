# merge_lora_and_save.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import shutil
from pathlib import Path

def lora_merge(base_model_path, lora_model_path):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype="auto", device_map="cpu")

    # Load and apply the LoRA adapter
    model = PeftModel.from_pretrained(model, lora_model_path)

    # Merge LoRA weights into the base model
    model = model.merge_and_unload()

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--lora_model_path", type=str)
    parser.add_argument('--delete', action='store_true')

    return parser.parse_args()

def main():
    args = parse_args()

    # Use pathlib for more robust path handling
    lora_path = Path(args.lora_model_path)
    merged_model_path = lora_path.parent / f"{lora_path.name}_merged"

    if args.delete:
        if merged_model_path.exists():
            shutil.rmtree(merged_model_path)
            print(f"Deleted: {merged_model_path}")
        else:
            print(f"Path does not exist: {merged_model_path}")
    else:
        model = lora_merge(args.base_model_path, args.lora_model_path)
        model.save_pretrained(merged_model_path)

        # Also copy the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        tokenizer.save_pretrained(merged_model_path)
        print(f"Model and tokenizer saved to: {merged_model_path}")

if __name__ == "__main__":
    main()