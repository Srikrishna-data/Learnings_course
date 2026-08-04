# Chapter 5 — Large Language Models: How They Work and How to Use Them Well

> **Part 5 — LLMs**  
> **Target time:** 12–16 hours  
> **Outcome:** You can explain the LLM lifecycle, control generation, select an adaptation approach, and design a production LLM call that is measurable, cost-aware, and safe.

## 1. What is an LLM?

A **large language model (LLM)** is usually a transformer neural network trained to predict the next token in text, code, or other serialized data. At a large enough scale and with post-training, this simple objective yields useful behaviors: writing, extracting information, translating, classifying, reasoning over supplied context, and requesting tools.

```text
prompt text ─► tokenizer ─► token IDs ─► transformer layers ─► next-token probabilities
                                                                      │
                                              decoding/sampling ◄─────┘
                                                                      │
                                                               generated text
```

### What it is not

An LLM is not a database, authority system, calculator, or guaranteed source of truth. It generates a likely continuation based on training and the current context. It can produce a convincing answer that has no supporting evidence. Application engineering supplies trusted data, tools, structured validation, permissions, and fallbacks.

### Problems LLMs solve

| Task | Why an LLM helps | Better alternative when applicable |
|---|---|---|
| Extract fields from messy text | tolerates many natural formats | deterministic parser for fixed format |
| Draft/summarize communication | fluent language transformation | template for repetitive exact copy |
| Answer private policy questions | understands retrieved passages | keyword search for simple lookup |
| Route/support classify | works with sparse labels and text | small classifier at high stable volume |
| Call tools in variable workflows | maps intent to typed operation | fixed workflow when path is known |

> **Core production principle:** An LLM should propose language or a bounded structured intent. Your application, not the model, controls data access and consequential actions.

---

## 2. Tokenization and vocabulary

### What is a token?

A **token** is a piece of input defined by a tokenizer. It might be a word, part of a word, punctuation, whitespace, or code fragment. Tokens are the units the model reads and generates. They are not reliably equivalent to words or characters.

```text
Text:       "unbelievable!"
Possible:   ["un", "believ", "able", "!"]
Token IDs:  [ ... integers determined by a model tokenizer ... ]
```

### Vocabulary

The **vocabulary** is the set of token pieces a model can directly represent. Tokenizers use a finite vocabulary plus algorithms such as byte-pair encoding or related subword methods to encode arbitrary text. This is why an unfamiliar name, code identifier, or non-English word may use many tokens.

### Why tokenization exists

Operating at whole-word level would create a huge vocabulary and fail on new words. Operating at character level creates very long sequences. Subword tokenization is a practical compromise.

### Token counting in production

Token count affects four things:

1. Model context limits.
2. Input and output billing for many providers.
3. Latency: longer prompts generally take longer to process.
4. Retrieval/context design.

```python
def planning_estimate_tokens(text: str) -> int:
    """Rough English planning estimate only; use the target tokenizer in production."""
    return max(1, len(text) // 4)

def reject_if_over_budget(prompt: str, max_input_tokens: int = 6000) -> None:
    if planning_estimate_tokens(prompt) > max_input_tokens:
        raise ValueError("Prompt exceeds application budget")
```

Use the exact tokenizer or provider usage fields for budgets and invoices. A character-based estimate is not safe for enforcement.

### Common mistakes

- Assuming “one token equals one word.”
- Sending entire chat histories or documents because the model has a large context window.
- Omitting output-token allowance when checking context capacity.
- Mixing an embedding model’s token rules with a generation model’s token rules.

---

## 3. Embeddings and context windows

### Embeddings

An **embedding** model maps text, image, or other content into a vector. Similar meanings tend to be near each other in vector space. Generation models use internal embeddings too, but application-level embeddings are commonly used for retrieval, clustering, and recommendations.

```text
policy documents ─► chunk + embed ─► vector store
user question ────► embed ──────────► retrieve relevant authorized chunks
                                             │
                                             ▼
                                   LLM answer with citations
```

Embeddings solve fresh/private knowledge retrieval better than trying to retrain a model for every document change. They do not replace metadata, keyword search, access control, or source validation.

### Context window

The **context window** is the maximum total number of input and generated output tokens a model can consider in one request. It includes system/developer instructions, user content, tool schemas/results, retrieved passages, conversation history, and output reservation.

```text
[system instructions][tool schemas][conversation][retrieved evidence][user question][reserved output]
|──────────────────────── total must fit model context window ────────────────────────|
```

Large context is valuable for long documents and complex tasks, but it has limitations:

- It costs more and usually increases latency.
- Relevant evidence can be diluted by irrelevant material.
- Permission filtering and provenance still matter.
- Context does not persist automatically between requests.

**Best practice:** retrieve a small, high-quality, permission-filtered evidence set; use summaries for conversational history; externalize large artifacts into storage and refer to IDs.

---

## 4. How an LLM is trained

### Stage 1: pretraining

Pretraining uses large corpora of token sequences. The model predicts a missing or next token and gradually learns language regularities, code patterns, facts represented in the data, and useful internal representations.

```text
raw corpora ─► filtering/deduplication ─► tokenize ─► next-token training ─► base model
```

This is resource-intensive: datasets, distributed GPU/accelerator clusters, networking, experimentation, safety work, and evaluation. Most application teams consume pretrained models rather than train one from scratch.

### Stage 2: supervised fine-tuning (SFT)

Instruction-response examples teach a base model to follow helpful formats and task patterns. This improves usability but can also import label bias, style artifacts, or unsafe behavior if the data is poor.

### Stage 3: preference optimization

Humans or carefully designed automated processes compare responses. Post-training aims to prefer helpful, safe, truthful, and policy-aligned behavior.

| Method | What it does | Practical interpretation |
|---|---|---|
| RLHF | learns a reward model/preferences, then optimizes policy | powerful but operationally complex |
| DPO | directly trains model to prefer chosen over rejected outputs | simpler preference-optimization formulation |
| RLAIF | uses AI-assisted feedback, usually with human calibration | can scale feedback but must be audited |

**RLHF** means reinforcement learning from human feedback. **DPO** (Direct Preference Optimization) uses preference pairs without the same explicit RL loop. Both depend heavily on data quality and evaluation; neither makes a model inherently factual or safe in every context.

### Training data and governance

Training/finetuning data needs provenance, rights review, privacy handling, quality checks, deduplication, and a clear retention/deletion policy. “It was in a production log” is not sufficient permission to train on it.

---

## 5. Generation controls: temperature, top-p, and top-k

At each generation step, the model produces a probability distribution over possible next tokens. A decoder selects one token and repeats.

### Temperature

**Temperature** reshapes the distribution before selection.

- Lower temperature: more repeatable, more concentrated choices.
- Higher temperature: more varied, more surprising choices.

Temperature does not turn uncertainty into factual certainty. For extraction, classification, and tool arguments, use low-variance decoding plus schema validation. For brainstorming, use controlled diversity and evaluation.

### Top-p and top-k

**Top-k** limits choices to the `k` highest-probability tokens. **Top-p** (nucleus sampling) limits choices to the smallest set whose cumulative probability reaches `p`.

| Control | Why it exists | Useful pattern | Caution |
|---|---|---|---|
| Temperature | control randomness | low for structured tasks | not a correctness setting |
| Top-k | restrict candidate count | constrained creative output | may remove viable rare tokens |
| Top-p | dynamically restrict probability mass | natural-language variety | interactions can be hard to predict |
| Max output tokens | cap response length/cost | all production requests | avoid truncating required JSON |

> **Best practice:** Change one decoding parameter at a time and compare on an evaluation set. Do not tune from a single impressive conversation.

### Production call configuration

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class GenerationPolicy:
    model: str
    temperature: float
    max_output_tokens: int
    timeout_seconds: float

EXTRACTION_POLICY = GenerationPolicy(
    model="approved-structured-model",
    temperature=0.0,
    max_output_tokens=350,
    timeout_seconds=20.0,
)
```

Record the model version, prompt version, decoding configuration, request IDs, input/output usage, latency, and validation result—not raw sensitive content by default.

---

## 6. Fine-tuning: when it helps and when it does not

Fine-tuning adapts model behavior using examples. It is useful for stable recurring patterns: company-specific labels, strict output style, specialized transformations, or domain language where a prompt baseline falls short.

```text
curated examples ─► quality/privacy review ─► train adapter/model ─► held-out eval ─► canary release
```

### Fine-tuning versus alternatives

| Need | Start with | Fine-tune when |
|---|---|---|
| Current/private facts | Retrieval (RAG) | almost never for facts alone |
| Exact application response | Schema + deterministic code | model behavior remains insufficient |
| Repeated classification/extraction | Prompted baseline / small ML | many quality examples justify it |
| Brand/style consistency | prompt + examples | long recurring prompts still underperform/cost too much |
| New capability | model/tool upgrade | task is stable and evaluation supports it |

### Limitations and costs

Fine-tuning does not guarantee that a model remembers every example, avoids hallucinations, or handles new policies. Costs include data preparation, training, evaluation, hosting, lifecycle management, regressions, and rollback—not only a training invoice.

### Common mistakes

- Fine-tuning to inject changing knowledge instead of using RAG.
- Training on unreviewed production logs containing PII or poor outcomes.
- Comparing a fine-tuned model only against a weak prompt baseline.
- Omitting a held-out evaluation set and a rollback model.

---

## 7. PEFT, LoRA, and QLoRA

### Parameter-efficient fine-tuning (PEFT)

Full fine-tuning updates all parameters. **PEFT** adapts a small subset of trainable parameters while keeping the base model mostly frozen. It lowers GPU memory and storage needs and makes it practical to maintain several task-specific adapters.

### LoRA

**Low-Rank Adaptation (LoRA)** adds small low-rank matrices to selected model layers. Instead of training a massive weight update directly, it learns a compact adjustment.

```text
base weight W (frozen) + low-rank update A × B (trained) = adapted behavior
```

### QLoRA

**QLoRA** combines quantized base-model weights with LoRA adapters, greatly reducing memory required for fine-tuning. It can make experiments feasible on more modest GPU hardware, but quality, throughput, hardware compatibility, and training stability must still be evaluated.

| Approach | Advantages | Limitations | Typical fit |
|---|---|---|---|
| Full fine-tuning | maximum flexibility | expensive memory/compute | large, mature model program |
| LoRA | cheap adapters, fast iteration | base-model constraints remain | domain/task adaptation |
| QLoRA | lower memory footprint | extra quantization trade-offs | budget-conscious experimentation |
| Prompt/RAG only | simplest operationally | may hit quality/cost ceiling | default first approach |

### Practical workflow

1. Create a versioned, de-identified task dataset.
2. Establish a prompt/RAG baseline.
3. Define success and safety metrics before training.
4. Train a small adapter; evaluate on untouched cases.
5. Test structured outputs, tool calls, edge cases, and latency.
6. Canary deploy with a fast rollback path.

---

## 8. MoE, quantization, and distillation

### Mixture of Experts (MoE)

An MoE model has multiple expert subnetworks and a router that activates only a subset per token. This can increase total parameter capacity without using every parameter for every token.

```text
token representation ─► router ─► expert 2 + expert 7 ─► combined output
```

**Advantages:** large capacity with lower active compute than a dense model of comparable total size.

**Limitations:** routing/load balancing, distributed serving, and capacity constraints add complexity. Do not assume a higher total parameter count alone predicts task quality.

### Quantization

Quantization stores/calculates weights with fewer bits (for example, 8-bit or 4-bit rather than higher precision). It reduces memory bandwidth and can enable smaller/cheaper hardware.

**Trade-off:** quality may degrade unevenly. Validate instruction following, multilingual behavior, structured output, and tool-call reliability—not only a generic benchmark.

### Distillation

Distillation trains a smaller **student** model to reproduce useful behavior from a larger **teacher** model. It can reduce latency and cost for a narrow workload.

```text
strong teacher outputs + labeled data ─► student training ─► evaluated smaller model
```

Distillation is worthwhile only when sustained volume and a stable task justify the data/training/maintenance investment.

---

## 9. Production LLM architecture

```text
Client
  │
  ▼
API / authentication ─► request validation ─► policy + budget ─► model gateway
      │                         │                    │               │
      │                         ▼                    │        route / retry / fallback
      └── audit + redacted tracing ◄─────────────────┘               │
                                                                    ▼
                                                        LLM + retrieval + typed tools
                                                                    │
                                              schema validation + citations + safe UI
```

### Required production controls

- Explicit model/prompt/configuration versions.
- Timeouts and limited retries only for safe/idempotent calls.
- Structured outputs validated by application code.
- Input size, output size, and per-user/tenant budgets.
- Retrieval and tool authorization outside the model.
- Evaluation gates before model or prompt releases.
- Fallback behavior: smaller model, deterministic path, escalation, or clear failure—not invented certainty.

### Cost and token optimization

| Lever | Benefit | Risk to manage |
|---|---|---|
| Shorter stable instructions | fewer input tokens | unclear task contract |
| Retrieval/reranking | avoids whole-document prompts | poor retrieval can hurt quality |
| Prompt caching where supported | lower repeated-prefix cost/latency | retention/privacy configuration |
| Output caps | controls runaway answers | truncated schema/output |
| Model routing | reserves strong models for hard tasks | router regression or inconsistent UX |
| Small fine-tuned/quantized model | lower recurring cost | quality and operational maintenance |

---

## 10. Hands-on project — structured invoice extraction

Build an LLM-backed service that extracts `vendor`, `invoice_date`, `currency`, `total`, `line_items`, and `needs_review` from invoice text or OCR output.

### Requirements

1. Define a Pydantic/JSON schema; do not parse prose with regex.
2. Make `needs_review` required for missing or conflicting fields.
3. Use a low-variance generation policy and an explicit output budget.
4. Validate outputs; reject malformed or impossible values (negative totals, invalid dates).
5. Create a 50-document evaluation set with difficult layouts and currencies.
6. Compare zero-shot, few-shot, retrieval of vendor rules, and a potential fine-tuning plan.
7. Track field accuracy, schema-valid rate, review rate, p95 latency, input/output tokens, and cost/document.
8. Redact/store documents according to a defined data-retention policy.

### Suggested result contract

```python
from datetime import date
from pydantic import BaseModel, Field

class InvoiceResult(BaseModel):
    vendor: str | None
    invoice_date: date | None
    currency: str | None
    total: float | None = Field(default=None, ge=0)
    line_items: list[str]
    needs_review: bool
```

### Interview questions

1. Why does temperature 0 not guarantee correct extraction?
2. RAG versus fine-tuning for a changing vendor policy—what would you select?
3. Explain LoRA and QLoRA to a platform engineer.
4. What information belongs in an LLM trace, and what should be redacted?
5. How would you evaluate whether quantization is acceptable?

---

## Summary

LLMs generate tokens using transformer networks trained through pretraining and post-training. Tokenization and context windows make input design a cost, latency, and quality concern. Temperature, top-p, and top-k shape diversity but do not create truth. Retrieval addresses current/private knowledge; fine-tuning changes stable behavior; LoRA/QLoRA make adaptation more efficient; MoE, quantization, and distillation trade capacity, cost, and complexity.

Reliable LLM applications use narrow contracts, trusted context, validated structured outputs, explicit budgets, evaluation, and safe fallbacks.

## Further reading

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)
- [OpenAI — Fine-tuning guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
- [DPO paper](https://arxiv.org/abs/2305.18290)

---

**Next chapter:** Part 6 — Generative AI Model Families: ChatGPT, Claude, Gemini, DeepSeek, Mistral, Llama, and Qwen; their trade-offs, selection process, and production routing.
