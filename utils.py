from __future__ import annotations
import time
import logging
from typing import Iterable
from vllm import LLM, SamplingParams

LOGGER = logging.getLogger(__name__)

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
            messages, tokenize=False, add_generation_prompt=True,
        )
    rendered = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
    rendered.append('Assistant:')
    return '\n'.join(rendered)


def load_vllm_llm(model_id, tensor_parallel_size: int = 1, **kwargs):
    """Load model with vLLM. Works on V100 32GB without quantization for 7B models."""
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
        dtype="auto",
        quantization=quantization,
        **kwargs,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def prompt_vllm(
    llm, tokenizer,
    batch_messages: Iterable[list[dict]],
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    top_p: float = 1.0,
    use_tqdm: bool = True,
):
    prompts = [build_vllm_prompt(tokenizer, messages) for messages in batch_messages]
    sampling_params = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    return [output.outputs[0].text if output.outputs else '' for output in outputs]


class TimeGuard:
    """Track elapsed time and stop before deadline."""
    def __init__(self, limit_minutes: int, safety_margin_minutes: int = 10):
        self.start = time.time()
        self.limit = limit_minutes * 60
        self.margin = safety_margin_minutes * 60

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> float:
        return self.limit - self.elapsed()

    def should_stop(self) -> bool:
        return self.remaining() < self.margin

    def log_status(self, logger):
        e = self.elapsed() / 60
        r = self.remaining() / 60
        logger.info("Time: %.1f min elapsed, %.1f min remaining", e, r)