import os
import sys
import argparse
from unittest.mock import MagicMock
import importlib.util

# --- Robust torchao bypass for Unsloth/transformers ---
def patch_torchao():
    try:
        import torchao  # real package present (Colab/GPU env) — skip mocking entirely
        return
    except ImportError:
        pass
    # torchao absent (local dev without GPU) — apply mock bypass so imports don't hard-fail
    for mod in ['torchao', 'torchao.quantization', 'torchao.prototype',
                'torchao.prototype.safetensors', 'torchao.prototype.safetensors.safetensors_support',
                'torchao.prototype.safetensors.safetensors_utils']:
        mock = MagicMock()
        mock.__spec__ = importlib.util.spec_from_loader(mod, loader=None)
        if mod == 'torchao': mock.__version__ = '0.17.0'
        if mod == 'torchao.quantization': mock.Float8Tensor = type(None)
        sys.modules[mod] = mock

patch_torchao()

os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "0.5")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments


def get_hardware_config() -> dict:
    if not torch.cuda.is_available():
        print("\n" + "="*50)
        print("[HARDWARE AUDIT] No CUDA device found — CPU fallback")
        print("[DEPLOYMENT] Hardware Tier: CONSTRAINED | Final Mode: CONSTRAINED | 4-bit: True")
        print("="*50 + "\n")
        return {"mode": "constrained", "use_4bit": True, "dtype": torch.float16}

    vram_gb      = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    compute_cap  = torch.cuda.get_device_capability(0)
    gpu_name     = torch.cuda.get_device_name(0).lower()
    bf16_capable = compute_cap[0] >= 8

    if vram_gb >= 32:
        hw_mode  = "fidelity"
        dtype    = torch.bfloat16 if bf16_capable else torch.float16
        use_4bit = False
    elif vram_gb >= 22:
        hw_mode  = "balanced"
        dtype    = torch.float16
        use_4bit = False
    else:
        hw_mode  = "constrained"
        dtype    = torch.float16
        use_4bit = True

    # Intent override: can only downgrade, never exceed hardware tier
    override   = os.environ.get("TRAINING_MODE", "").lower()
    tier_rank  = {"constrained": 1, "balanced": 2, "fidelity": 3}
    final_mode = hw_mode
    if override in tier_rank and tier_rank[override] < tier_rank[hw_mode]:
        final_mode = override
        if final_mode == "constrained":
            use_4bit = True
            dtype    = torch.float16

    print("\n" + "="*50)
    print(f"[HARDWARE AUDIT] GPU: {gpu_name} | VRAM: {vram_gb:.1f}GB | BF16: {bf16_capable}")
    print(f"[DEPLOYMENT] Hardware Tier: {hw_mode.upper()} | Final Mode: {final_mode.upper()} | 4-bit: {use_4bit}")
    print("="*50 + "\n")
    return {"mode": final_mode, "use_4bit": use_4bit, "dtype": dtype}


def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT Training Entrypoint")
    parser.add_argument('--dataset', type=str, default='output/datasets/theorist/combined_theorist_v2.jsonl', help='Path to input dataset (JSONL)')
    parser.add_argument('--output', type=str, default='runs/theorist_model', help='Output directory for GGUF model')
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = args.output
    mode_name = "theorist_v2"

    print(f"\n{'='*50}\n[FORGING] Starting training for mode: {mode_name.upper()}\n{'='*50}")

    hw_config  = get_hardware_config()
    model_name = (
        "unsloth/Meta-Llama-3.1-8B-Instruct"
        if not hw_config["use_4bit"]
        else "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = 2048,
        dtype          = hw_config["dtype"],
        load_in_4bit   = hw_config["use_4bit"],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = True,
    )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    def format_prompts(examples):
        texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"] if isinstance(msg, list)]
        # Fallback for text-based mapping if ChatML is pre-formatted
        if not texts and "text" in examples:
            return {"text": examples["text"]}
        return {"text": texts}

    if "messages" in dataset.column_names:
        dataset = dataset.map(format_prompts, batched=True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 1,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = 2,
            learning_rate = 1.5e-5,
            fp16 = not (hw_config["dtype"] == torch.bfloat16 and torch.cuda.is_bf16_supported()),
            bf16 = hw_config["dtype"] == torch.bfloat16 and torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"runs/lora_checkpoints/{mode_name}",
        ),
    )

    trainer.train()

    print(f"[merge] Merging LoRA into 16-bit model in-memory...")
    from unsloth.save import unsloth_save_model as _save_merged, save_to_gguf as _save_to_gguf

    _save_merged(
        model=model,
        tokenizer=tokenizer,
        save_directory=output_dir,
        save_method="merged_16bit",
        max_shard_size="2GB",
        maximum_memory_usage=0.85,
    )

    print(f"[gguf] Converting to Q4_K_M GGUF...")
    _save_to_gguf(
        model_name=mode_name,
        model_type="llama",
        model_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        is_sentencepiece=False,
        model_directory=output_dir,
        quantization_method="q4_k_m",
        first_conversion="f16",
    )

if __name__ == "__main__":
    main()
