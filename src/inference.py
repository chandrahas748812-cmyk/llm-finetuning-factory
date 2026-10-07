"""Inference with a fine-tuned LoRA adapter (or base model).

Usage:
    python -m src.inference --base-model meta-llama/Meta-Llama-3-8B \\
        --adapter ./registry/meta-llama-meta-llama-3-8b/v1/adapter \\
        --prompt "Explain LoRA in one sentence."

Without --adapter, runs the base model. Requires torch + transformers + peft.
A MockBackend is provided for offline smoke tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class MockBackend:
    """Deterministic offline backend for smoke tests."""

    def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        return f"[mock] completion for: {prompt[:80]}"


def load_backend(base_model: str, adapter: str | None):
    if adapter is None:
        raise RuntimeError("No adapter and no mock requested")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    class HFBackend:
        def generate(self, prompt: str, max_new_tokens: int = 128) -> str:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            import torch as _t

            with _t.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            return tokenizer.decode(out[0], skip_special_tokens=True)

    return HFBackend()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    backend = MockBackend() if args.mock else load_backend(args.base_model, args.adapter)
    print(backend.generate(args.prompt, args.max_new_tokens))


if __name__ == "__main__":
    main()
