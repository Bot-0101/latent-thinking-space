# Q-COMMIT-B — does output-simplex curvature predict CoT correctness beyond entropy?
# Colab / Jupyter notebook as ordered `# %%` cells. Subject: Qwen2.5-1.5B-Instruct (HF transformers). Seed 42.
# Design + pre-registration: experiments/q-commit-b/problem-statement.md
# NOTE: runs on Colab GPU (not locally). Cells flagged [RUNTIME-VERIFY] have one API/shape detail to eyeball on first run.

# %% [0] setup ---------------------------------------------------------------
# !pip install -q transformers datasets accelerate
import torch, numpy as np, random, json, os, re
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
RESULTS = "results"; os.makedirs(RESULTS, exist_ok=True)

# Signal B needs only OUTPUT next-token distributions -> plain transformers (no nnsight; Signal A was dropped).
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", torch_dtype=torch.float32).eval()
print("vocab_size:", model.config.vocab_size)   # EXPECT 151936

# %% [1] SANITY / tokenizer-first (repo rule: a silent format bug corrupts everything) -------
# Print the chat-templated prompt + a sample tokenization; confirm where the assistant turn starts;
# reproduce a trivial known behaviour.
SYS = "You are a helpful assistant. Solve the problem step by step, then give the final answer as 'The answer is N.'"
def build_prompt(question):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

demo_q = "Natalia sold clips to 48 friends in April, and half as many in May. How many did she sell altogether?"
p = build_prompt(demo_q)
print("=== TEMPLATED PROMPT ===\n", p)
print("=== LAST 12 TOKENS (confirm it ends at <|im_start|>assistant\\n) ===")
ids = tok(p, return_tensors="pt")["input_ids"][0]
print([tok.decode([t]) for t in ids[-12:]])

# trivial known behaviour: greedy-generate a short answer, eyeball coherence
with torch.no_grad():
    o = model.generate(**tok(p, return_tensors="pt").to(model.device), max_new_tokens=80,
                       do_sample=False, pad_token_id=tok.eos_token_id)
print("=== SAMPLE GENERATION ===\n", tok.decode(o[0][ids.shape[0]:], skip_special_tokens=True))
# STOP-CHECK: read the above. If the template/answer looks wrong, fix before proceeding.

# %% [2] data + answer parsing ----------------------------------------------
from datasets import load_dataset
gsm = load_dataset("gsm8k", "main")["test"].select(range(100))   # first 100 test, seed-independent slice

def gold_answer(a):                       # gold format: "... #### 72"
    m = re.search(r"####\s*([-0-9,\.]+)", a); return m.group(1).replace(",", "").strip() if m else None
def parse_pred(text):                     # robust: model uses LaTeX \boxed{}, echoes "The answer is N.", etc.
    text = re.sub(r"\\boxed\{([^{}]*)\}", r" \1 ", text)      # unwrap \boxed{50} -> 50
    NUM = r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?"                   # MUST start with a digit (no lone '.' / ',')
    m = re.findall(rf"answer\s*(?:is|=|:)\s*\$?\(?\s*({NUM})", text, flags=re.I)  # prefer explicit marker
    if not m:  m = re.findall(NUM, text)                      # fallback: last real number in the text
    return m[-1].replace(",", "").rstrip(".") if m else None
def is_correct(pred, gold):
    try: return pred is not None and gold is not None and abs(float(pred) - float(gold)) < 1e-6
    except ValueError: return False

# %% [3] generate CoT + capture the RAW per-step next-token distributions -----
# transformers generate(output_scores=True) returns one [1,vocab] logit tensor per token ACTUALLY generated
# (stops at EOS cleanly, no phantom iterations). Under greedy (no warpers) scores==logits, so softmax = the dist.
MAXNEW = 512   # was 320; verbose LaTeX CoT overflowed it -> truncated tails -> mislabeled wrong
@torch.no_grad()
def run_one(question):
    inputs = tok(build_prompt(question), return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAXNEW, do_sample=False,
                         output_scores=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
    gen_ids = out.sequences[0][inputs["input_ids"].shape[1]:]
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)
    probs = [torch.softmax(s[0].float(), dim=-1).cpu().numpy() for s in out.scores]  # one (vocab,) per step
    return probs, gen_text

from geometry import geom_features, entropy_features   # compute per-question, then DISCARD probs (memory)
records = []
for i, ex in enumerate(gsm):
    probs, gen = run_one(ex["question"])               # probs ~150 MB (≈250 steps x 152k vocab) -- TRANSIENT
    pred, gold = parse_pred(gen), gold_answer(ex["answer"])
    records.append(dict(i=i, gen=gen, pred=pred, gold=gold, correct=is_correct(pred, gold),
                        geom=geom_features(probs), ent=entropy_features(probs)))  # keep only small features
    del probs                                          # free the big array NOW (was the RAM blow-out ~Q40)
    if i % 20 == 0: print(f"{i}/100 | acc-so-far {np.mean([r['correct'] for r in records]):.2f}")
print("overall accuracy:", np.mean([r["correct"] for r in records]))

# %% [4] geometry + baseline + stats — imported from geometry.py -------------
# Pure functions live in geometry.py so they're unit-testable without the model.
# Verified locally by test_geometry.py: 21/21 PASS (Fisher-Rao, Menger, entropy, spearman, primary+null).
from geometry import d_fr, menger_curv, geom_features, entropy_features, spearman, bootstrap_diff

# %% [5] filter to usable questions (features already computed per-question in cell 3; no raw probs kept) ----
valid = [r for r in records if r["geom"] and r["ent"]]
print("usable questions:", len(valid))

# %% [6] PRIMARY test — PRE-REGISTERED single feature each (1-vs-1, unbiased by feature count) ----
# Dry-run fix: "best |rho| over k columns" gave the side with more features an unfair edge (and re-opened
# the cherry-picking risk the audit flagged). Pre-registered primary scalars instead:
#   curvature = curv_per_len (arc-length-normalised total curvature) ;  entropy = mean_H (mean uncertainty).
y = np.array([r["correct"] for r in valid], dtype=float)
g = np.array([r["geom"]["curv_per_len"] for r in valid])
e = np.array([r["ent"]["mean_H"] for r in valid])
rho_curv, rho_ent = abs(spearman(g, y)), abs(spearman(e, y))
ci, mean_diff = bootstrap_diff(g, e, y, seed=SEED, n=2000)
print(f"PRIMARY | rho_curv={rho_curv:.3f}  rho_ent={rho_ent:.3f}  diff={mean_diff:.3f}  CI95={ci}")
print("FALSIFIER: curvature adds signal only if CI lower bound > 0.")

# EXPLORATORY only (NOT the falsifier) — for /analyze, never to rescue a null primary:
for k in ("mean_curv", "max_curv"):
    print(f"  explor rho({k})={abs(spearman([r['geom'][k] for r in valid], y)):.3f}")
for k in ("slope_H", "frac_decreasing", "final_H"):
    print(f"  explor rho({k})={abs(spearman([r['ent'][k] for r in valid], y)):.3f}")

# %% [7] BASELINE tier-2 (expensive, faithful 2603.18940 on a 30-Q subset) ---
# Their metric: at each CoT step prefix, sample m=5 completions (tau=0.7, max 150), entropy over final-answer freqs,
# then binary eps-monotone (eps=0.01; any step with H_{k+1} > H_k + eps => non-monotone). Accuracy gap monotone vs not.
SUBSET, M, EPS = 30, 5, 0.01
def step_prefixes(gen):                   # segment CoT into step prefixes (their regex, with fallback)
    marks = [m.start() for m in re.finditer(r"Step\s*\d+:", gen)]
    if len(marks) < 2: marks = [m.start() for m in re.finditer(r"\n\n", gen)]
    return [gen[:m] for m in marks] or [gen[:len(gen)//2]]

@torch.no_grad()
def answer_entropy_at(question, prefix):
    inputs = tok(build_prompt(question) + prefix, return_tensors="pt").to(model.device)
    ans = []
    for _ in range(M):
        o = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7,
                           pad_token_id=tok.eos_token_id)
        ans.append(parse_pred(tok.decode(o[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)))
    c = Counter(ans); tot = sum(c.values())
    return -sum((n/tot) * np.log(n/tot) for n in c.values())

t2 = []
for r in valid[:SUBSET]:
    Hs = [answer_entropy_at(gsm[r["i"]]["question"], pf) for pf in step_prefixes(r["gen"])]
    monotone = all(Hs[k+1] <= Hs[k] + EPS for k in range(len(Hs)-1)) if len(Hs) > 1 else True
    t2.append(dict(i=r["i"], monotone=monotone, correct=r["correct"], mean_curv=r["geom"]["mean_curv"]))
mono = [d["correct"] for d in t2 if d["monotone"]]; non = [d["correct"] for d in t2 if not d["monotone"]]
print(f"TIER-2 2603.18940 baseline | monotone acc={np.mean(mono) if mono else float('nan'):.2f} (n={len(mono)}) "
      f"vs non-monotone acc={np.mean(non) if non else float('nan'):.2f} (n={len(non)})")

# %% [8] save metrics + plots + RANDOM (not cherry-picked) examples ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
summary = dict(seed=SEED, n=len(valid), accuracy=float(y.mean()),
               rho_curv=rho_curv, rho_ent=rho_ent, primary_diff=mean_diff, primary_CI95=list(ci),
               tier2_monotone_acc=float(np.mean(mono)) if mono else None,
               tier2_nonmonotone_acc=float(np.mean(non)) if non else None)
json.dump(summary, open(f"{RESULTS}/summary.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(["curv", "ent"], [rho_curv, rho_ent]); ax[0].set_title("best |Spearman rho| vs correctness")
ax[1].scatter(g, y + np.random.uniform(-0.03, 0.03, len(y)), s=12); ax[1].set_xlabel("curv_per_len (pre-reg)"); ax[1].set_ylabel("correct")
plt.tight_layout(); plt.savefig(f"{RESULTS}/primary.png", dpi=120)

rng = np.random.default_rng(SEED)                     # RANDOM examples for /analyze — do NOT cherry-pick
sample = [valid[k] for k in rng.choice(len(valid), size=min(8, len(valid)), replace=False)]
json.dump([dict(i=r["i"], correct=r["correct"], pred=r["pred"], gold=r["gold"],
                mean_curv=r["geom"]["mean_curv"], mean_H=r["ent"]["mean_H"], gen=r["gen"]) for r in sample],
          open(f"{RESULTS}/random_examples.json", "w"), indent=2)
print("saved:", os.listdir(RESULTS))

# %% [9] HANDOFF ------------------------------------------------------------
# Do NOT interpret here. Numbers are in results/summary.json; random examples in results/random_examples.json.
# Next: /analyze (fresh skeptical pass). A first positive result is probably an artifact until falsified.
print(json.dumps(summary, indent=2))
