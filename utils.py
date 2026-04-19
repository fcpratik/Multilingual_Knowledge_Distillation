from __future__ import annotations
import os
import subprocess
import glob
from typing import Iterable


# ══════════════════════════════════════════════════════════════
# Kaggle / Colab environment fixes — must run BEFORE vllm import
# ══════════════════════════════════════════════════════════════

# Fix 1: Create libcuda.so symlink (FlashInfer linker needs it)
# Kaggle has libcuda.so.1 but not libcuda.so, and stubs/ may be read-only
def _fix_libcuda():
    """Find libcuda.so.1 and ensure libcuda.so exists somewhere on LD path."""
    # Find the actual libcuda
    candidates = glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so*") + \
                 glob.glob("/usr/local/cuda/lib64/libcuda.so*") + \
                 glob.glob("/usr/lib64/libcuda.so*")

    source = None
    for c in candidates:
        if os.path.exists(c) and not os.path.islink(c):
            source = c
            break
        elif os.path.exists(c):
            source = c
            break

    if source is None:
        return

    # Try multiple writable locations
    link_targets = [
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/lib/libcuda.so",
    ]

    for target in link_targets:
        if os.path.exists(target):
            return  # Already exists
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.symlink(source, target)
            # Add to library path if non-standard location
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            target_dir = os.path.dirname(target)
            if target_dir not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = f"{target_dir}:{ld_path}"
            return
        except OSError:
            continue


_fix_libcuda()

# Fix 2: Disable FlashInfer entirely — it needs JIT compilation
# which fails on Kaggle due to missing libcuda.so in stubs.
# Force vLLM to use V0 engine which supports XFORMERS/TRITON backends.
os.environ["VLLM_USE_V1"] = "0"  # Force V0 engine
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"  # V0 supports this

from vllm import LLM, SamplingParams

MMLU_TEMPLATE = '''Answer the following multiple choice question. The last line of your response should be of the following format: '#### ANSWER: [LETTER]' (without quotes) where [LETTER] is one of {letters}. Think step by step before answering.

{question}

{choices}'''
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def format_mmlu(question, choices):
    choices_str = '\n'.join(
        f'({letter}) {choice}' for letter, choice in zip(LETTERS, choices)
    )
    return MMLU_TEMPLATE.format(question=question, choices=choices_str, letters=LETTERS[:len(choices)])


def build_vllm_prompt(tokenizer, messages):
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].capitalize()}: {message['content']}")
    rendered.append('Assistant:')
    return '\n'.join(rendered)


def load_vllm_llm(model_id, tensor_parallel_size: int = 1, **kwargs):
    # Determine if this is a quantized model by checking the name
    quantization = None
    if "AWQ" in model_id or "awq" in model_id:
        quantization = "awq"
    elif "GPTQ" in model_id or "gptq" in model_id:
        quantization = "gptq"

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="half",
        enforce_eager=True,
        quantization=quantization,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def prompt_vllm(
        llm,
        tokenizer,
        batch_messages: Iterable[list[dict]],
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_tqdm: bool = True,
):
    prompts = [build_vllm_prompt(tokenizer, messages) for messages in batch_messages]
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    return [output.outputs[0].text if output.outputs else '' for output in outputs]