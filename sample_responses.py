import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import os
from utils import format_prompt, load_prompts, lora_merge_and_load_vllm
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--batch_size", type=int) 
    parser.add_argument("--max_new_tokens", type=int, default=2048) 
    parser.add_argument("--temperature", type=float, default=1.0) 
    parser.add_argument("--top_p", type=float, default=1.0) 
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.lora_path is None:
        print("evaluating a non-lora model")
        if not os.path.exists(args.model_path):
            if "finetuned_models" in args.model_path:
                raise NotImplementedError
                # os.makedirs(args.model_path.replace('/', '_'), exist_ok=True)
                # output_path = os.path.join(args.model_path.replace('/', '_'), "strongreject_responses.json")
                # metadata_path = os.path.join(args.model_path.replace('/', '_'), "strongreject_sampling_params.json")
            else:
                model_path_full = os.path.join("./finetuned_models", args.model_path.replace('/', '_'), "base")
                os.makedirs(model_path_full, exist_ok=True)
                output_path = os.path.join(model_path_full, "strongreject_responses.json")
                metadata_path = os.path.join(model_path_full, "strongreject_sampling_params.json")
        else:
            output_path = os.path.join(args.model_path, "strongreject_responses.json")
            metadata_path = os.path.join(args.model_path, "strongreject_sampling_params.json")
    else:
        print("evaluating a lora model")
        output_path = os.path.join(args.lora_path, "strongreject_responses.json")
        metadata_path = os.path.join(args.lora_path, "strongreject_sampling_params.json")


    print(f"Responses will be saved to {output_path}")

    if args.dataset_name == "walledai/StrongREJECT":
        prompt_field = "prompt"  
        split = "train"
    else:
        raise NotImplementedError
    
    # === Load dataset ===
    all_prompts = load_prompts(args.dataset_name, split, prompt_field)

    # === Load model + tokenizer ===
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        num_gpus = len(visible_devices)
    else:
        num_gpus = torch.cuda.device_count()  # falls back to all available GPUs

    print(f"Using {num_gpus} GPUs for tensor parallelism")

    if args.lora_path is None:
        llm = LLM(model=args.model_path, dtype="auto", tensor_parallel_size=num_gpus)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        llm = lora_merge_and_load_vllm(args.model_path, args.lora_path, tensor_parallel_size=num_gpus)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # === Sampling params ===
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens
    )

    # === Batched inference and save ===
    responses = {}
    for i in tqdm(range(0, len(all_prompts), args.batch_size)):
        batch = all_prompts[i:i + args.batch_size]
        prompts = [format_prompt(tokenizer, p) for p in batch]
        outputs = llm.generate(prompts, sampling_params)

        for p, output in zip(batch, outputs):
            response_text = output.outputs[0].text.strip()
            responses[p] = response_text
        
        if args.debug:
            print(responses)
            break

    # === Save responses to JSON ===
    with open(output_path, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    # === Save sampling parameters to separate JSON ===
    metadata = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {len(responses)} responses to {output_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
