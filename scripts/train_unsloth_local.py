

# --- Robust torchao bypass for Unsloth/transformers ---
import os
import sys
import argparse
from unittest.mock import MagicMock
import importlib.util

def patch_torchao():
    torchao_mock = MagicMock()
    torchao_mock.__spec__ = importlib.util.spec_from_loader('torchao', loader=None)
    torchao_mock.__version__ = '0.17.0'
    torchao_mock.__path__ = []
    sys.modules['torchao'] = torchao_mock

    quantization_mock = MagicMock()
    quantization_mock.__spec__ = importlib.util.spec_from_loader('torchao.quantization', loader=None)
    # Float8Tensor must be a real type so isinstance(W, Float8Tensor) doesn't raise TypeError
    quantization_mock.Float8Tensor = type(None)
    sys.modules['torchao.quantization'] = quantization_mock

    prototype_mock = MagicMock()
    prototype_mock.__spec__ = importlib.util.spec_from_loader('torchao.prototype', loader=None)
    prototype_mock.__path__ = []
    sys.modules['torchao.prototype'] = prototype_mock

    safetensors_mock = MagicMock()
    safetensors_mock.__spec__ = importlib.util.spec_from_loader('torchao.prototype.safetensors', loader=None)
    safetensors_mock.__path__ = []
    sys.modules['torchao.prototype.safetensors'] = safetensors_mock

    safetensors_support_mock = MagicMock()
    safetensors_support_mock.__spec__ = importlib.util.spec_from_loader('torchao.prototype.safetensors.safetensors_support', loader=None)
    sys.modules['torchao.prototype.safetensors.safetensors_support'] = safetensors_support_mock

    safetensors_utils_mock = MagicMock()
    safetensors_utils_mock.__spec__ = importlib.util.spec_from_loader('torchao.prototype.safetensors.safetensors_utils', loader=None)
    sys.modules['torchao.prototype.safetensors.safetensors_utils'] = safetensors_utils_mock

patch_torchao()

# On Windows WDDM, torch.cuda.mem_get_info() can report 0 free even when the GPU
# can still serve allocations via system-RAM overflow.  Unsloth's CE-loss guard
# hard-fails at 0, so we override it with a fixed 0.5 GB target (chunked CE loss).
os.environ.setdefault("UNSLOTH_CE_LOSS_TARGET_GB", "0.5")
# expandable_segments lets the CUDA allocator grow beyond the reported free
# memory by using WDDM-backed virtual address space (system RAM overflow).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch._inductor.config
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments


def patch_safetensors_no_mmap():
    """
    Windows paging-file workaround (OSError 1455 / ERROR_PAGEFILE_QUOTA).

    safetensors.safe_open memory-maps the file and Windows must reserve paging-file
    space equal to the entire file before any page is read — crashing immediately on
    systems with small paging files.

    Strategy: lazy binary reader.  Each _RamSafeOpen keeps the file handle open but
    only reads a tensor when __getitem__ is called on its slice.  This means at most
    one tensor worth of bytes lives in RAM at a time during loading, and the file-
    mapping API is never touched.  Thread-safe via a per-file lock (transformers can
    materialise tensors from a ThreadPoolExecutor).
    """
    import struct, json, threading, mmap as _mmap
    import numpy as np
    import transformers.modeling_utils as _mu
    import safetensors.torch as _st
    import safetensors as _saf

    _DTYPE_MAP = {
        "F32":  torch.float32,  "F16":  torch.float16,  "BF16": torch.bfloat16,
        "I8":   torch.int8,     "I16":  torch.int16,    "I32":  torch.int32,
        "I64":  torch.int64,    "U8":   torch.uint8,    "BOOL": torch.bool,
        "F64":  torch.float64,
    }
    _NP_MAP = {
        torch.float32: "float32", torch.float16: "float16",
        torch.int8:    "int8",    torch.int16:   "int16",
        torch.int32:   "int32",   torch.int64:   "int64",
        torch.uint8:   "uint8",   torch.float64: "float64", torch.bool: "bool",
    }

    class _LazySlice:
        """Returned by get_slice(); reads bytes from the parent file on first access."""
        __slots__ = ("_parent", "_meta")
        def __init__(self, parent, meta):
            self._parent = parent
            self._meta   = meta

        def get_shape(self):  return list(self._meta["shape"])
        def get_dtype(self):  return self._meta["dtype"]   # e.g. "F16", "BF16", "U8"

        def __getitem__(self, _idx):
            return self._parent._load(self._meta)

    # Chunk size for reading large tensors without a single large allocation
    _CHUNK_BYTES = 64 * 1024 * 1024   # 64 MB per chunk

    class _RamSafeOpen:
        """Drop-in for safetensors.safe_open.

        Reads tensors in 64 MB chunks via plain fh.read() to avoid both
        Windows pagefile mmap failures and single large RAM allocations.
        For tensors that land on CUDA (the common path), chunks are streamed
        directly into a pre-allocated VRAM buffer so peak CPU RAM is only
        64 MB per tensor regardless of tensor size.
        """
        def __init__(self, path, framework="pt", device="cpu"):
            self._device = device
            self._lock   = threading.Lock()
            self._fh     = open(path, "rb")
            hlen         = struct.unpack("<Q", self._fh.read(8))[0]
            header       = json.loads(self._fh.read(hlen))
            self._ds     = 8 + hlen
            self._hdr    = {k: v for k, v in header.items() if k != "__metadata__"}

        def __del__(self):
            try: self._fh.close()
            except Exception: pass

        def __enter__(self): return self
        def __exit__(self, *_):
            try: self._fh.close()
            except Exception: pass

        def keys(self):              return self._hdr.keys()
        def offset_keys(self):       return self._hdr.keys()   # same order, for load_file
        def get_slice(self, key):    return _LazySlice(self, self._hdr[key])
        def get_tensor(self, key):   return self._load(self._hdr[key])

        def _load(self, meta):
            off_s, off_e = meta["data_offsets"]
            tdtype  = _DTYPE_MAP.get(meta["dtype"], torch.float32)
            shape   = meta["shape"]
            nbytes  = off_e - off_s
            item_sz = torch.tensor([], dtype=tdtype).element_size()
            n_elem  = nbytes // item_sz

            # Detect CUDA availability — even if self._device=="cpu" transformers
            # will immediately move the tensor to CUDA via _materialize_copy.
            # We stream directly to a VRAM buffer to keep CPU RAM <= 64 MB.
            cuda_ok = torch.cuda.is_available()

            if cuda_ok and nbytes > _CHUNK_BYTES:
                # --- CUDA path: stream chunks into pre-allocated VRAM buffer ---
                result = torch.empty(n_elem, dtype=tdtype, device="cuda")
                elem_per_chunk = _CHUNK_BYTES // item_sz
                written = 0
                with self._lock:
                    self._fh.seek(self._ds + off_s)
                    while written < n_elem:
                        take     = min(elem_per_chunk, n_elem - written)
                        raw      = self._fh.read(take * item_sz)
                        if tdtype == torch.bfloat16:
                            chunk = torch.frombuffer(bytearray(raw),
                                                     dtype=torch.uint16
                                                     ).view(torch.bfloat16)
                        else:
                            chunk = torch.frombuffer(bytearray(raw), dtype=tdtype)
                        result[written : written + take].copy_(chunk)
                        written += take
                t = result
            else:
                # --- CPU path: single read (tensor is small or no CUDA) ---
                with self._lock:
                    self._fh.seek(self._ds + off_s)
                    raw = self._fh.read(nbytes)
                if tdtype == torch.bfloat16:
                    t = torch.frombuffer(bytearray(raw),
                                         dtype=torch.uint16).view(torch.bfloat16).clone()
                else:
                    t = torch.frombuffer(bytearray(raw), dtype=tdtype).clone()

            if shape:
                t = t.reshape(shape)
            # If transformers asked for cpu but we stored on cuda, it will
            # call .to("cpu") / .to("cuda:0") itself — that's fine.
            return t

    _mu.safe_open  = _RamSafeOpen
    _st.safe_open  = _RamSafeOpen
    _saf.safe_open = _RamSafeOpen
    # Also patch saving_utils which uses `from safetensors import safe_open` (local binding)
    try:
        import unsloth_zoo.saving_utils as _sutils
        _sutils.safe_open = _RamSafeOpen
    except Exception:
        pass
    print("[patch] safe_open -> lazy binary reader (Windows paging-file bypass)")

def main():
    parser = argparse.ArgumentParser(description="Unsloth SFT Training Entrypoint")
    parser.add_argument('--dataset', type=str, required=True, help='Path to input dataset (JSONL)')
    parser.add_argument('--output', type=str, required=True, help='Output directory for GGUF model')
    parser.add_argument('--gguf-only', action='store_true', help='Skip training; load checkpoint and convert to GGUF only')
    parser.add_argument('--checkpoint', type=str, default=None, help='LoRA checkpoint dir to load for --gguf-only')
    parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint in lora_checkpoints/{mode_name}')
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = args.output
    mode_name = os.path.splitext(os.path.basename(output_dir))[0]

    print(f"\n{'='*50}\n[FORGING] Starting training for mode: {mode_name.upper()}\n{'='*50}")
    patch_safetensors_no_mmap()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
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

    if args.gguf_only:
        checkpoint = args.checkpoint or f"output/lora_checkpoints/{mode_name}/checkpoint-174"
        print(f"[GGUF-ONLY] Loading checkpoint: {checkpoint}")
        from peft import set_peft_model_state_dict
        import safetensors.torch
        state = safetensors.torch.load_file(f"{checkpoint}/adapter_model.safetensors")
        set_peft_model_state_dict(model, state)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        def format_prompts(examples):
            texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in examples["messages"]]
            return {"text": texts}
        dataset = dataset.map(format_prompts, batched=True)
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = 1024,
            dataset_num_proc = 1,
            args = TrainingArguments(
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs = 2,
                learning_rate = 1.5e-5,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = f"output/lora_checkpoints/{mode_name}",
            ),
        )
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        free_b, total_b = torch.cuda.mem_get_info(0)
        stats = torch.cuda.memory_stats(0)
        alloc_b = stats.get("allocated_bytes.all.current", 0)
        resv_b  = stats.get("reserved_bytes.all.current", 0)
        print(f"[VRAM] free={free_b/1e9:.2f}GB  alloc={alloc_b/1e9:.2f}GB  reserved={resv_b/1e9:.2f}GB  total={total_b/1e9:.2f}GB")
        resume_ckpt = True if args.resume else None
        trainer.train(resume_from_checkpoint=resume_ckpt)
        del trainer

    os.makedirs(output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # Install gguf package if absent — required by the convert_hf_to_gguf subprocess
    try:
        import gguf as _gguf_test; del _gguf_test
    except ImportError:
        print("[patch] Installing gguf Python package for GGUF conversion...")
        import subprocess as _sp
        _sp.run([sys.executable, "-m", "pip", "install", "gguf", "--quiet"], check=True)

    # Capture model config before freeing the model
    _model_type  = getattr(model.config, "model_type", "llama")
    _model_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"

    # transformers 5.5.0 added revert_weight_conversion which calls reverse_op on
    # each operation; _IdentityOp (used for dequantize pass-through) doesn't implement
    # reverse_op and raises NotImplementedError.  Patch it to a safe fallback.
    try:
        import transformers.core_model_loading as _cml
        import transformers.modeling_utils as _mu2
        _orig_revert = _cml.revert_weight_conversion
        def _safe_revert(model, state_dict):
            try:
                return _orig_revert(model, state_dict)
            except (NotImplementedError, ValueError):
                return state_dict if isinstance(state_dict, dict) else dict(state_dict)
        _cml.revert_weight_conversion = _safe_revert
        _mu2.revert_weight_conversion = _safe_revert
        print("[patch] revert_weight_conversion -> safe fallback (transformers 5.5.0 compat)")
    except Exception as _e:
        print(f"[patch] revert_weight_conversion patch skipped: {_e}")

    # transformers 5.5.0 save_pretrained: when hf_quantizer is present, it replaces
    # the passed state_dict with hf_quantizer.get_state_dict_and_metadata() which
    # returns raw BnB weights (base_layer.weight keys) — ignoring our merged dict.
    # Fix: null out hf_quantizer temporarily when save_pretrained is given an explicit
    # state_dict (the Unsloth merge path always passes one).
    try:
        from transformers.modeling_utils import PreTrainedModel as _PTM
        _orig_sp = _PTM.save_pretrained
        def _patched_sp(self, *_a, state_dict=None, **_kw):
            if state_dict is not None:
                _q = getattr(self, "hf_quantizer", None)
                self.hf_quantizer = None
                try:
                    return _orig_sp(self, *_a, state_dict=state_dict, **_kw)
                finally:
                    if _q is not None:
                        self.hf_quantizer = _q
            else:
                return _orig_sp(self, *_a, **_kw)
        _PTM.save_pretrained = _patched_sp
        print("[patch] save_pretrained -> hf_quantizer bypass for explicit state_dict")
    except Exception as _e:
        print(f"[patch] save_pretrained patch skipped: {_e}")

    # Use the OLD in-memory merge path (unsloth_save_model) to avoid the new
    # merge_and_overwrite_lora path which downloads 16 GB of FP16 weights and
    # then hits the Windows paging-file limit in Rust safetensors.
    print(f"[merge] Merging LoRA into 16-bit model in-memory (no FP16 download)...")
    from unsloth.save import unsloth_save_model as _save_merged, save_to_gguf as _save_to_gguf
    _save_merged(
        model=model,
        tokenizer=tokenizer,
        save_directory=output_dir,
        save_method="merged_16bit",
        max_shard_size="2GB",
        maximum_memory_usage=0.85,
    )

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # _download_convert_hf_to_gguf downloads a fresh copy of convert_hf_to_gguf.py
    # and tries to exec it for introspection, but the latest version references
    # gguf.MODEL_ARCH.GEMMA4 / MISTRAL4 etc. that don't exist in the pip gguf.
    # Fix: replace the function with one that applies the gguf attr patches locally
    # to the cached converter and returns that path, skipping the download.
    try:
        import re as _re
        import unsloth.save as _us
        import unsloth_zoo.llama_cpp as _llc

        _CONV_DIR = _llc.LLAMA_CPP_DEFAULT_DIR
        _RAW      = os.path.join(_CONV_DIR, "convert_hf_to_gguf.py")
        _PATCHED  = os.path.join(_CONV_DIR, "unsloth_convert_hf_to_gguf.py")

        def _local_patch_converter():
            if not os.path.exists(_PATCHED):
                with open(_RAW, "rb") as _fh:
                    _content = _fh.read()
                # Find all gguf.X.Y.Z references and add try/except guards
                _refs = list(set(_re.findall(rb"[\n\s]gguf\.([\.A-Z\_0-9]{3,})[\n\s\,]", _content)))
                _refs = [x.decode("utf-8") for x in _refs if not x.startswith(b"_")]
                if _refs:
                    _edits = "\n".join(
                        f"try: gguf.{x}\nexcept AttributeError: gguf.{x} = None"
                        for x in _refs
                    ).encode("utf-8")
                    _content = _re.sub(
                        rb"(import gguf\s*\n)", rb"\1" + _edits + b"\n\n",
                        _content, count=1
                    )
                with open(_PATCHED, "wb") as _fh:
                    _fh.write(_content)
                print(f"[patch] wrote patched converter -> {_PATCHED}")
            return _PATCHED, None, None

        _us._download_convert_hf_to_gguf = _local_patch_converter
        print("[patch] _download_convert_hf_to_gguf -> local gguf-attr patcher")
    except Exception as _e:
        print(f"[patch] _download_convert_hf_to_gguf patch skipped: {_e}")

    print(f"[gguf] Converting {mode_name} to Q4_K_M GGUF (q8_0 intermediate)...")
    _save_to_gguf(
        model_name=mode_name,
        model_type=_model_type,
        model_dtype=_model_dtype,
        is_sentencepiece=False,
        model_directory=output_dir,
        quantization_method="q4_k_m",
        first_conversion="f16",
    )

if __name__ == "__main__":
    main()
