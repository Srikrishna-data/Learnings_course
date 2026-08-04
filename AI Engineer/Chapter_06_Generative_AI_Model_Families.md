# Chapter 6 — Generative AI Model Families: Selecting GPT, Claude, Gemini, DeepSeek, Mistral, Llama, and Qwen

> **Part 6 — Generative AI**  
> **Target time:** 8–10 hours  
> **Outcome:** You can compare model families without relying on hype, build a reproducible model evaluation, and route tasks according to quality, latency, cost, privacy, and operational constraints.

## 1. Why model selection is an engineering decision

Generative-AI model names change quickly. A good choice is therefore not a permanent declaration that one vendor “wins.” It is a repeatable decision process: define the task, evaluate approved current models on representative cases, apply security and deployment constraints, and select the lowest-cost option that clears quality and reliability thresholds.

```text
Product requirement ─► security / legal constraints ─► evaluation set
                                                        │
                                                        ▼
                       current candidate models ─► quality + latency + cost scorecard
                                                        │
                                                        ▼
                                              route, release, monitor, re-evaluate
```

### Chat applications versus model APIs

**ChatGPT**, **Claude**, and **Gemini** are widely used product brands as well as model ecosystems. Engineers should distinguish the consumer experience from an API model release. A model used in a chat product can have different tools, system behavior, usage limits, retention terms, and versions from the model exposed through an API.

> **Callout — never hardcode a marketing claim.** Context limits, pricing, availability, tool support, and model versions are time-sensitive. Fetch approved current metadata from a provider API or official documentation during procurement and release reviews.

---

## 2. The major model ecosystems

### GPT and OpenAI ecosystem

GPT models are accessed through OpenAI’s developer platform and are commonly chosen for managed general-purpose generation, reasoning, structured outputs, tools, multimodal workflows, audio, and coding. The platform also provides product-level capabilities such as hosted tools and evaluation-related features.

**Advantages:** mature managed platform, broad modality/tool ecosystem, fast startup experience, and current models discoverable through official model documentation/API.

**Limitations:** managed-service dependency, provider-specific abstractions, variable availability by region/account, and data/control requirements that must be reviewed per endpoint and contract.

**Typical fit:** startups moving quickly; teams that need high-quality managed APIs; products that benefit from integrated multimodal/tool functionality.

### Claude and Anthropic ecosystem

Claude models are Anthropic’s managed model family. They are commonly evaluated for language, document analysis, coding, and tool-using workflows. Anthropic exposes models through its direct API and supported cloud platforms.

**Advantages:** strong managed option for long-form language workloads, direct API, and platform ecosystem.

**Limitations:** same managed-vendor considerations: availability, pricing, limits, and data controls must be verified for the particular deployment path.

**Typical fit:** document-heavy workflows, coding evaluation candidates, and teams whose approved cloud/provider options include Anthropic.

### Gemini and Google ecosystem

Gemini models are Google’s generative model family available through developer and cloud paths. The official API supports programmatic model discovery, including metadata such as supported functions and token limits.

**Advantages:** broad multimodal orientation, Google-cloud integration options, and model metadata available via API.

**Limitations:** product/API/cloud paths can differ; regional, billing, governance, and feature availability need an explicit evaluation.

**Typical fit:** teams already operating on Google Cloud, multimodal candidates, and applications where the Google ecosystem is an operational advantage.

### Llama ecosystem

Llama is Meta’s model family. Several releases are distributed as downloadable/open-weight models under Meta’s license terms rather than as a single universally “open source” offering. Meta documents text and multimodal variants and an ecosystem of local/cloud deployment options.

**Advantages:** deployment control, broad serving ecosystem, customization experimentation, and a path for private-cloud or local inference.

**Limitations:** the user owns serving, security patches, GPU capacity, observability, and model lifecycle. License terms and exact model capabilities must be reviewed for each release.

**Typical fit:** enterprises requiring deployment control, teams with ML platform capacity, and workloads whose sustained volume justifies self-hosting analysis.

### Mistral ecosystem

Mistral provides managed APIs and publishes selected model weights. Its catalog spans model sizes and deployment patterns.

**Advantages:** choice between provider-managed and selected self-managed/open-weight paths; useful candidate where regional/deployment requirements or model-size options matter.

**Limitations:** model features and license/hosting options vary by release; self-hosting remains an operational commitment.

**Typical fit:** organizations wanting optionality between managed and self-hosted approaches, subject to task evaluation and commercial review.

### DeepSeek ecosystem

DeepSeek models are commonly evaluated in the open-weight and hosted-API landscape, including reasoning and coding-oriented workloads depending on the release.

**Advantages:** a valuable benchmark candidate for cost-conscious or self-hosting-oriented evaluations; active ecosystem interest.

**Limitations:** verify exact licensing, hosted-service terms, data residency, support model, security posture, model weights provenance, and deployment requirements. “Available weights” does not remove governance obligations.

**Typical fit:** controlled benchmark and private-deployment evaluations, especially where the organization can operate the model responsibly.

### Qwen ecosystem

Qwen is an Alibaba-developed family with text, code, vision/language, and smaller-model variants across releases.

**Advantages:** broad size/modality ecosystem and a strong candidate for multilingual, self-hosted, or small-model evaluations depending on the current version.

**Limitations:** licensing, supported languages, infrastructure fit, and quality are release-specific. Validate the exact weights and serving stack.

**Typical fit:** multilingual and open-weight evaluations; constrained deployments where small models may be attractive.

---

## 3. Closed models, open weights, and managed hosting

“Open source,” “open weight,” and “self-hosted” are not synonyms.

| Approach | What you receive | Advantages | Costs / limitations |
|---|---|---|---|
| Managed API | model endpoint run by provider | fastest launch, no GPU operations | vendor dependency, recurring API spend, policy/data review |
| Open-weight model | downloadable trained weights under a license | deployment/customization control | must operate inference, evaluate security/license, fund GPUs |
| Fully open stack | weights plus code/data details to a defined degree | research/transparency benefits | still requires operations and governance |
| Hosted open-weight model | provider serves an open-weight family | easier operations with more choice | combines provider dependency with release/license complexity |

### Startup versus enterprise defaults

| Concern | Startup default | Enterprise pattern |
|---|---|---|
| Time to product | managed API | managed API or approved cloud marketplace |
| Model diversity | one or two evaluated providers | gateway with approved provider portfolio |
| Data governance | provider DPA/settings review | DLP, regional routing, audit, private networking, contractual controls |
| Serving | provider-hosted | self-host only for measured privacy/volume/control needs |
| Reliability | provider fallback + escalation | multi-region, multi-provider or private fallback where justified |

The enterprise pattern is not automatically better. It adds platform and operational cost. A small team should not operate GPU clusters merely to avoid a few API calls if the economics and compliance requirements do not justify it.

---

## 4. How to compare models correctly

### Define the task before the model

Bad requirement: “Find the best LLM.”  
Good requirement: “Extract six invoice fields from English and Spanish PDFs, with ≥98% schema-valid responses, ≥95% exact total accuracy, p95 under 4 seconds, and ≤$0.03 per reviewed document.”

### Scorecard dimensions

| Dimension | How to measure | Why it matters |
|---|---|---|
| Task quality | blinded labeled cases, human calibration | benchmarks alone rarely match your task |
| Reasoning | multi-step correct-answer cases with evidence | visible fluency is not correctness |
| Coding | run generated patches through tests/lint/security review | prose explanations are insufficient |
| Structured output | schema-valid rate and field accuracy | tool integrations depend on this |
| Function calling | valid arguments, correct tool selection, safe abstention | prevents unsafe tool automation |
| Multimodal | labeled image/audio/document cases | modality claims vary by model and API |
| Speed | time-to-first-token and p50/p95 end-to-end | user experience and capacity |
| Cost | input + output + tools + hosting per successful task | token price alone is incomplete |
| Context | retrieval/faithfulness at realistic document lengths | advertised window ≠ useful recall |
| Governance | region, retention, DPA, access logs, license | can rule out an otherwise strong model |

### A fair model bake-off

```text
representative production-like cases
             │
             ├─► candidate A ─► validated result ─┐
             ├─► candidate B ─► validated result ─┼─► blind grading / metrics
             └─► candidate C ─► validated result ─┘
                                                         │
                                       cost + latency + safety + operability
                                                         │
                                                  release recommendation
```

Use the same prompt contract, context, output schema, timeout, and tool simulation where possible. Record model ID/version, region, date, SDK version, retry behavior, and all parameters. Compare candidates on the same cases, preferably with graders who do not know which model produced the answer.

### Example score computation

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelScore:
    name: str
    quality: float              # 0 to 1 from a representative evaluation
    p95_seconds: float
    cost_per_task: float
    schema_valid_rate: float

def eligible(m: ModelScore) -> bool:
    return (
        m.quality >= 0.92
        and m.schema_valid_rate >= 0.99
        and m.p95_seconds <= 5.0
        and m.cost_per_task <= 0.05
    )

def choose(models: list[ModelScore]) -> ModelScore:
    options = [m for m in models if eligible(m)]
    if not options:
        raise RuntimeError("No model satisfies the release gate")
    return min(options, key=lambda m: m.cost_per_task)
```

This avoids the common mistake of picking the model with the highest raw quality when a nearly equal model is faster, cheaper, and more reliable for the task.

---

## 5. Model routing and fallback design

One product can use more than one model. A **model gateway** is an internal service that centralizes authentication, logging, routing, quotas, fallback rules, and provider abstraction.

```text
application ─► model gateway ─► policy / budget / task classifier
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
           small fast model        strong model             self-hosted model
           simple extraction       complex analysis          restricted workload
                  │                       │                       │
                  └────────────────── validated response ──────────┘
```

### Routing examples

| Task | Appropriate route pattern |
|---|---|
| Language detection, short classification | small/fast model or classical ML |
| Simple stable extraction | lower-cost model + strict schema |
| Contract analysis with citations | stronger evaluated model + RAG + review |
| Private restricted workload | approved private endpoint or self-hosted model |
| Provider outage | pre-evaluated alternate or safe escalation |

### Fallback principles

Fallback must be designed and evaluated before an outage. A fallback to a weaker model may be unsafe for a high-risk workflow. In that case the correct fallback is often: show a temporary-unavailable message, preserve the request, and send it to a human queue.

```python
def resolve_route(task: str, risk: str, provider_healthy: bool) -> str:
    if risk == "high" and not provider_healthy:
        return "human_escalation"      # safer than an unqualified substitute
    if task == "classification":
        return "small_model"
    return "primary_strong_model" if provider_healthy else "evaluated_fallback"
```

### Cost implications

Model routing can reduce spend substantially by reserving stronger models for difficult cases. It also introduces router errors, behavior inconsistency, test complexity, and governance work. Track cost and quality by route; do not assume a routing change saves money until it preserves outcome metrics.

---

## 6. Provider integration patterns

### Keep provider code behind an interface

Your product should not expose every provider SDK throughout the codebase. Define a narrow internal contract for the feature you need.

```python
from typing import Protocol

class TextModel(Protocol):
    def generate_json(self, *, prompt: str, schema: dict, timeout_s: float) -> dict: ...

def extract_invoice(client: TextModel, prompt: str, schema: dict) -> dict:
    raw = client.generate_json(prompt=prompt, schema=schema, timeout_s=20)
    # Validate raw against application schema here; never trust provider output blindly.
    return raw
```

This makes a provider migration or comparison possible without rewriting feature logic. Do not hide meaningful differences, though: capability flags, modality support, rate limits, and data controls should be explicit configuration.

### Production checklist

- Pin an approved model version/snapshot where available; record the date when not.
- Use an explicit timeout, bounded safe retries, and circuit breaker.
- Enforce per-tenant quotas and token budgets at the gateway.
- Validate structured outputs in application code.
- Redact or minimize sensitive telemetry; configure provider retention appropriately.
- Run regression evals before any provider/model upgrade.
- Maintain a deprecation/migration plan.

---

## 7. Common mistakes and interview questions

### Common mistakes

| Mistake | Better approach |
|---|---|
| Selecting a vendor from a public leaderboard | run a task-specific, versioned evaluation |
| Comparing price per token only | compare cost per successful task, including retries/tools/human review |
| Calling all downloadable models “open source” | read the exact license and deployment terms |
| Using a model’s large context instead of retrieval | evaluate evidence quality, permissions, and cost |
| Failing over dynamically without tests | qualify every fallback before production |
| Assuming chat and API behavior are identical | evaluate the exact API model/configuration |

### Interview questions

1. How would you run a fair comparison of GPT, Claude, Gemini, and an open-weight model for customer-support extraction?
2. What factors beyond quality determine whether a model is production-ready?
3. Explain managed API versus self-hosted open weights to a finance leader.
4. Why might a model with the longest context window still need RAG?
5. Design a safe fallback when the primary model provider is unavailable.
6. How would you prevent a model upgrade from silently regressing a tool-calling workflow?

---

## 8. Hands-on project — reproducible model bake-off

Build a model comparison harness for a fictional operations assistant.

### Deliverables

1. Write 60 representative cases: extraction, short classification, cited Q&A, and tool selection.
2. Create a strict response schema and deterministic validators.
3. Implement provider adapters behind one internal interface.
4. Run at least three approved candidates on the identical dataset.
5. Measure quality, schema-valid rate, time-to-first-token, p95 end-to-end latency, tokens, cost/task, and safe-abstention behavior.
6. Add a blind human review sample and calibrate any LLM grader against it.
7. Document provider/model IDs, dates, regions, and configuration.
8. Create a decision record that explains selection, fallback, and re-evaluation cadence.

### Architecture

```text
evaluation dataset ─► runner ─► adapter A / B / C ─► raw traces (redacted)
                              │                              │
                              ▼                              ▼
                      deterministic validators         human/LLM grading
                              └───────────────► scorecard + decision record
```

## Summary

GPT, Claude, Gemini, DeepSeek, Mistral, Llama, and Qwen should be understood as evolving ecosystems rather than permanent rankings. Managed APIs optimize time-to-value; open-weight deployment can optimize control or sustained-volume economics but creates an infrastructure responsibility. The durable practice is a versioned, task-specific scorecard that weighs quality, speed, cost, context usefulness, function-calling reliability, governance, and operational risk.

## Further reading

- [OpenAI model documentation](https://developers.openai.com/api/docs/models/all)
- [Anthropic model API](https://platform.claude.com/docs/en/api/models/list)
- [Gemini models documentation](https://ai.google.dev/gemini-api/docs/models)
- [Meta Llama documentation](https://ai.meta.com/llama/get-started/)
- [Mistral model API](https://docs.mistral.ai/api/endpoint/models)

---

**Next chapter:** Part 7 — Prompt Engineering: zero-shot and few-shot prompts, instruction hierarchy, XML/JSON, structured outputs, tool/function calling, and evaluation-driven prompt design.
