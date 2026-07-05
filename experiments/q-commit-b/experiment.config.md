# experiment.config — q-commit-b

| field | value |
|-------|-------|
| **library** | **HF transformers** (Signal A was dropped, so no internal hooks are needed — Signal B uses only OUTPUT distributions via `generate(output_scores=True)`. nnsight/TransformerLens not required. Switched from nnsight after `tracer.iter[:]` proved fragile on early-EOS generations.) |
| **compute** | Colab (default). GPU T4/A100; 1.5B fp32 fits easily. |
| **model** | `Qwen/Qwen2.5-1.5B-Instruct` (open weights — **no HF_TOKEN needed**) |
| **seed** | 42 |
| **created** | 2026-07-05 |
| **design** | see `problem-statement.md` (pre-registered) and `research-loop/queue.md` → BUILD DECISION (Q-COMMIT-B) |

## Setup snippet (Colab)
```python
!pip install -q transformers datasets accelerate
import torch, numpy as np, random
from transformers import AutoModelForCausalLM, AutoTokenizer
random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)

M = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, device_map="auto", torch_dtype=torch.float32).eval()
print("vocab:", model.config.vocab_size)   # expect 151936
```

## Implementation notes (resolved)
- **Per-step next-token distribution during generation:** `model.generate(..., output_scores=True, return_dict_in_generate=True)` → `out.scores` is one `[1, vocab]` logit tensor per token actually generated (clean stop at EOS, no phantom iterations); `softmax(scores)` = the distribution under greedy decoding. Replaced the fragile nnsight `tracer.iter[:]` capture.
- **Tokenizer-first (repo rule) = slice 1:** print `tok.apply_chat_template(msgs, add_generation_prompt=True)` and a sample tokenization; confirm where the assistant turn / first generated token lands **before** any geometry.
- Save all plots to `results/`; persist to Drive so they survive the Colab session.

## Save plots / outputs → `results/`
