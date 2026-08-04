# Chapter 8 — AI Agents: Tools, Memory, Orchestration, and Reliable Workflows

> **Part 8 — AI Agents**  
> **Target time:** 12–16 hours  
> **Outcome:** You can decide when an agent is justified, design a constrained tool-using workflow, distinguish memory types, use MCP safely, and choose an orchestration framework deliberately.

## 1. What is an AI agent?

An AI agent is a system that uses a model to choose one or more next steps toward a goal. It commonly observes state, calls approved tools, evaluates the result, and either continues, asks for clarification, or stops.

```text
goal + current state ─► model/planner ─► approved tool ─► observation
          ▲                 │                                  │
          └──── checkpoint ─┴──── budget / policy / evaluator ──┘
```

This definition is intentionally narrower than marketing language. An agent is not simply a chat interface. It is not inherently autonomous, reliable, or safe. Its value comes from coordinating variable sequences of steps where a fixed workflow would be awkward.

### Agentic AI versus deterministic workflow

| Pattern | Who decides the next step? | Best for | Main risk |
|---|---|---|---|
| Deterministic workflow | application code | known business process | rigid on unexpected inputs |
| LLM-assisted workflow | code chooses flow; LLM fills a task | extraction, drafting, classification | model output quality |
| Agent | model selects among bounded actions | research, diagnosis, variable tool sequence | loops, unsafe action, cost/latency |

> **Start with a workflow.** Use an agent only when the path truly varies and the benefit of flexible tool selection exceeds the added risk, latency, and operational complexity.

### Real-world example

An operations assistant receives: “Why is order ORD-103 late, and what should I tell the customer?” A fixed workflow may always look up order → carrier → policy → draft reply. An agent may be justified only if it must decide whether to inspect inventory, carrier incident data, address exceptions, or a return policy. Even then, each lookup is a narrowly scoped tool, and sending a message requires human approval.

---

## 2. The agent loop and its boundaries

### Basic loop

```text
1. Receive goal and authenticated user context
2. Read current durable state
3. Choose a permitted next action or final response
4. Validate / authorize the action
5. Execute tool and record observation
6. Update checkpoint and budget
7. Stop, continue, or request human approval
```

### Minimal implementation sketch

```python
from dataclasses import dataclass, field

@dataclass
class RunState:
    goal: str
    steps: list[str] = field(default_factory=list)
    tool_calls: int = 0
    approved: bool = False

MAX_TOOL_CALLS = 4

def can_continue(state: RunState) -> bool:
    return state.tool_calls < MAX_TOOL_CALLS

def safe_next_step(state: RunState) -> str:
    if not can_continue(state):
        return "stop_and_escalate"
    # In a real application: model proposes an action; code validates it.
    return "lookup_order"
```

The crucial behavior is not the loop syntax. It is the constraints: time, tokens, tool calls, data access, action approvals, and a safe failure outcome.

### Required budgets

| Budget | Why it exists | Example |
|---|---|---|
| Tool turns | prevents loops and surprise actions | max 4 calls/run |
| Token budget | controls latency and cost | 12,000 total tokens |
| Wall-clock time | avoids hung jobs | 90 seconds interactive, longer async |
| Money/value limit | limits business impact | no refund above $0 without approval |
| Data scope | stops broad exfiltration | current tenant + current user’s records |

### Common mistakes

- Giving an agent unrestricted shell, SQL, browser, or network access.
- Sending full tool results back into context when a compact summary/ID is sufficient.
- Letting it retry write actions without idempotency protection.
- Building multi-agent coordination before a one-agent or deterministic path has been evaluated.

---

## 3. Tool calling: the bridge to real systems

### What it is

Tool calling lets a model request a typed operation. The model emits an intent and arguments. The application validates the schema, checks authorization, runs the operation, and supplies a limited result.

```text
model request ─► schema validation ─► RBAC / tenant check ─► service/database
      ▲                                                               │
      └──────────────── compact, redacted tool result ───────────────┘
```

### Tool design principles

1. **Narrow purpose:** `get_order_status`, not `run_any_sql`.
2. **Strict arguments:** enums, patterns, numeric limits, and required fields.
3. **Server-side authorization:** tool execution validates user/tenant; the model never decides permission.
4. **Read/write distinction:** write actions need confirmation, idempotency, audit events, and often human approval.
5. **Small response:** return only data necessary for the next step.

```python
from pydantic import BaseModel, Field

class RefundDraftArgs(BaseModel):
    order_id: str = Field(pattern=r"^ORD-[0-9]+$")
    reason: str = Field(max_length=300)

def create_refund_draft(args: RefundDraftArgs, *, tenant_id: str, user_id: str) -> dict:
    # Real service checks that this user can access this order and creates no payment.
    return {"draft_id": "draft_123", "status": "awaiting_human_approval"}
```

### Advantages and limitations

Tools ground an agent in source-of-truth data and real operations. They also expand the attack surface: a manipulated prompt may persuade the model to request a dangerous but schema-valid tool action. Policy, authorization, confirmation, and audit controls remain mandatory.

---

## 4. MCP: Model Context Protocol

### What it is

The Model Context Protocol (MCP) is a protocol for connecting AI applications to external tools, resources, and prompts through standardized client/server interactions. It can reduce bespoke integration work and lets one client work with multiple compatible servers.

```text
AI application / MCP client ─► MCP server ─► approved external system
                                      │
                              tools / resources / prompts
```

### Why it exists

Without a common protocol, every model application implements one-off adapters for files, databases, ticketing systems, source control, and internal APIs. MCP standardizes the connection shape; it does **not** make an external server trustworthy.

### Production security model

| Control | Why it matters |
|---|---|
| Server allowlist and ownership review | MCP servers can expose data/actions |
| Least-privilege credentials | limit blast radius |
| Explicit user consent | make data sharing and actions visible |
| Tool schema review | reject broad/dangerous operations |
| Egress and audit logging | detect misuse and support investigation |
| Version pinning / supply-chain review | avoid unexpected server changes |

> **MCP is an integration protocol, not a permission model.** Treat MCP servers like third-party code with access to your environment.

### When to use it

Use MCP when you need standardized, reviewed integrations across agent clients or developer tools. For a single internal microservice call, a direct typed API can be simpler and safer.

---

## 5. Planning, reflection, and human oversight

### Planning

Planning decomposes a goal into steps. It can improve complex task completion but consumes tokens and can make a system look productive while doing needless work. Keep plans explicit, short, and reviewable for costly actions.

```text
Goal: prepare incident update
Plan: (1) fetch incident facts; (2) check status page; (3) draft update; (4) request approval
```

### Reflection

Reflection asks a model or evaluator to review an intermediate result. It can catch errors, but a model may confidently validate its own mistake. Use deterministic validators, independent evidence, and human review for high-stakes outputs.

| Review type | Best use | Limitation |
|---|---|---|
| Schema/rule validator | format, ranges, permissions | cannot judge nuanced quality |
| Separate model judge | scalable subjective checks | needs calibration against humans |
| Retrieval/evidence check | citations and support | source quality may be poor |
| Human approval | consequential actions | adds cost and latency |

### Human-in-the-loop patterns

```text
agent prepares draft ─► human sees evidence + proposed action ─► approve / edit / reject
                                                                    │
                                                                    ▼
                                                             audited execution
```

Use approval for money movement, external communication, data deletion, security operations, legal/medical decisions, or any action where an error is hard to reverse.

---

## 6. Memory: short-term, long-term, and retrieval

“Memory” is overloaded. Separate the types explicitly.

| Type | What it contains | Storage | Example |
|---|---|---|---|
| Short-term memory | current conversation/run state | request/checkpoint store | current order ID and tool observations |
| Long-term memory | curated facts across sessions | governed profile/memory store | preferred language with consent |
| Retrieval | external source material fetched on demand | search/vector/database | current return policy |
| Artifact state | large documents, code, files | object store / database | generated report draft |

### Short-term memory

Keep only what the current run needs. Summarize old conversation turns; preserve source IDs and important decisions. A long raw transcript increases token cost and can distract the model.

### Long-term memory

Long-term memory should be explicit, editable, permissioned, and auditable. Do not infer or retain sensitive personal facts merely because a user mentioned them once. Define scope, retention, deletion, and access rules.

### Retrieval is not memory

Retrieval fetches authoritative, versioned knowledge. It should be preferred for policies and facts that change. A user preference store is not a substitute for a document index, and an embedding index is not a durable workflow state store.

### Token optimization

```text
raw history ─► summarize verified decisions ─► store large artifacts externally
                                                 │
next turn ─► retrieve only relevant state + evidence ─► model
```

This reduces cost and improves clarity—but summaries must not discard safety-critical or contractual information.

---

## 7. Orchestration and durable execution

### What orchestration solves

An agent can run for seconds, minutes, or longer. Network calls fail, workers restart, humans delay approvals, and jobs need to resume. Orchestration records state and controls transitions so a workflow is durable rather than an in-memory loop.

```text
start ─► classify ─► retrieve ─► tool call ─► approval? ─► final response
              │          │             │           │
              └──────── checkpoints / retries / audit ───────────────┘
```

### Production requirements

- Persist checkpoints after meaningful state changes.
- Make side-effecting actions idempotent.
- Separate synchronous API intake from long-running workers/queues.
- Use retry policies that distinguish transient failures from validation/policy failures.
- Support cancellation, expiry, escalation, and replayable traces.
- Use a relational database or durable state service for run history; do not rely only on chat history.

### Framework guide

| Option | Best fit | Strength | Caution |
|---|---|---|---|
| Plain Python + queue | simple bounded workflow | few dependencies, clear control | build persistence/observability yourself |
| LangGraph | stateful durable agent graph | explicit state, interrupts, persistence ecosystem | learn graph/state model first |
| CrewAI | role-oriented multi-agent prototype | quick role/task abstractions | ensure roles map to measurable need |
| AutoGen | agent conversation patterns | flexible multi-agent experimentation | can hide complex control flow |
| PydanticAI | typed Python agent apps | schemas and Python ergonomics | still need operational design |
| OpenAI Agents SDK | OpenAI-aligned agent primitives | tools, handoffs, tracing ecosystem | evaluate provider coupling |
| Temporal / workflow engine | mission-critical durable jobs | retries, signals, long-running state | not an LLM framework; adds platform work |

Frameworks do not provide security or correctness automatically. They organize control flow; your team still owns tool permissions, tests, data handling, and costs.

---

## 8. Single-agent and multi-agent systems

### Single agent

A single agent with specialized tools and deterministic substeps is easier to test, trace, and secure. It should be the default.

### Multi-agent systems

Multi-agent systems assign separate planners, researchers, coders, reviewers, or domain specialists. They are justified when subproblems are genuinely independent or permissions/responsibilities must be separated.

```text
coordinator ─► researcher ─► source bundle
      │                              │
      ├────► analyst ─────► findings ┼──► synthesizer ─► reviewed answer
      │                              │
      └────► policy checker ─────────┘
```

### Costs and limitations

Each additional agent increases prompt tokens, execution time, traces, failure modes, and coordination ambiguity. A “reviewer agent” can repeat the same hallucination unless it has independent tools, validators, or criteria. Use multi-agent design only when its measured task benefit exceeds these costs.

### Interview questions

1. When should a workflow replace an agent?
2. How would you prevent a tool-using agent from issuing duplicate refunds?
3. Explain short-term memory, long-term memory, and retrieval.
4. Why is MCP not a security boundary?
5. How would you resume an agent after a worker crash?
6. What evidence would justify a multi-agent architecture?

---

## 9. Hands-on project — approval-gated order operations agent

Build an agentic workflow that investigates order issues and prepares—but never directly executes—customer-facing actions.

### Required capabilities

1. Authenticate the user and carry tenant/user identity in state.
2. Read order status through a narrow tool with server-side ownership checks.
3. Retrieve the current refund/shipping policy with citations.
4. Create a structured action draft with rationale, source IDs, and risk level.
5. Pause for simulated human approval before sending a message or creating a refund request.
6. Persist state after each tool call; support resuming after a deliberate failure.
7. Enforce limits: four tool calls, 90 seconds, 12,000 tokens, and no unapproved write.
8. Test prompt injection in retrieved policy text, unauthorized order access, tool-loop limit, provider timeout, and duplicate action requests.

### Suggested architecture

```text
Web/API ─► auth ─► agent orchestrator ─► checkpoint database
                         │       │
                    retrieval    approved tools ─► order / policy services
                         │       │
                         └── traces + budgets + audit ─► monitoring
                                      │
                                      ▼
                                  approval inbox
```

### Deliverables

- Architecture decision record: why agent versus workflow.
- Threat model and tool-permission table.
- 40-case evaluation suite and cost/latency report.
- Runbook for provider failure and stuck approval.

---

## Summary

AI agents are bounded systems that select steps and use tools toward a goal. Their benefits come with nondeterminism, cost, latency, and security risks. Start with deterministic workflows, introduce an agent only for variable paths, design narrow tools with server-side authorization, use MCP as a reviewed integration protocol, separate memory from retrieval, and persist state for durable execution. Single-agent systems are the default; multi-agent systems require evidence that separation improves the task.

## Further reading

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [PydanticAI](https://ai.pydantic.dev/)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Temporal durable execution](https://docs.temporal.io/encyclopedia/durable-execution)
- [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/)

---

**Next chapter:** Part 9 — Retrieval-Augmented Generation (RAG): chunking, embeddings, vector databases, hybrid search, reranking, citations, and production evaluation.
