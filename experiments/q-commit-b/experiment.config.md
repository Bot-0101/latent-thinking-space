# experiment.config — q-commit-b

| field | value |
|-------|-------|
| **library** | **nnsight** (TransformerLens does NOT support Qwen2.5 — verified via `loading_from_pretrained.py`; nnsight wraps the HF model directly) |
| **compute** | Colab (default). GPU T4/A100; 1.5B fp32 fits easily. |
| **model** | `Qwen/Qwen2.5-1.5B-Instruct` (open weights — **no HF_TOKEN needed**) |
| **seed** | 42 |
| **created** | 2026-07-05 |
| **design** | see `problem-statement.md` (pre-registered) and `research-loop/queue.md` → BUILD DECISION (Q-COMMIT-B) |

## Setup snippet (Colab)
```python
!pip install -q nnsight
import torch, numpy as np, random
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

from nnsight import LanguageModel
model = LanguageModel("Qwen/Qwen2.5-1.5B-Instruct", device_map="auto", torch_dtype=torch.float32)
tok = model.tokenizer
print("vocab:", model.config.vocab_size)   # expect 151936
```

## Implementation notes / open API checks (resolve at /implement — DO NOT invent)
- **Verify via nnsight docs (context7 / nnsight.net):** the exact pattern to capture the **per-step next-token softmax distribution during generation** (`model.generate(...)` tracing + saving `logits`/`lm_head` output per decoded token). This is the one API detail not yet confirmed — look it up, don't guess.
- **Tokenizer-first (repo rule) = slice 1:** print `tok.apply_chat_template(msgs, add_generation_prompt=True)` and a sample tokenization; confirm where the assistant turn / first generated token lands **before** any geometry.
- Save all plots to `results/`; persist to Drive so they survive the Colab session.

## Save plots / outputs → `results/`
