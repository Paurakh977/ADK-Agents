# ADK Streaming vs Non-Streaming: Complete Runtime Guide

This directory demonstrates two modes of agent execution using Google's Agent Development Kit (ADK): **streaming** (`main.py`) and **non-streaming** (`non_streaming.py`). Before understanding streaming, you need to understand the runtime that powers both.

---

## Table of Contents

- [Part 1: The ADK Runtime Foundation](#part-1-the-adk-runtime-foundation)
  - [What Is the Runtime?](#what-is-the-runtime)
  - [The Event Loop: Core Mechanism](#the-event-loop-core-mechanism)
  - [Key Runtime Components](#key-runtime-components)
  - [Runner: The Orchestrator](#runner-the-orchestrator)
  - [Event: The Message Protocol](#event-the-message-protocol)
  - [RunConfig: Controlling Runtime Behavior](#runconfig-controlling-runtime-behavior)
  - [SessionService & Session](#sessionservice--session)
  - [A Full Invocation Walkthrough](#a-full-invocation-walkthrough)
- [Part 2: Streaming vs Non-Streaming](#part-2-streaming-vs-non-streaming)
  - [What `event.partial` Actually Means](#what-eventpartial-actually-means)
  - [The Two-Level Streaming Switch](#the-two-level-streaming-switch)
  - [Streaming Mode (`main.py`)](#streaming-mode-mainpy)
  - [Non-Streaming Mode (`non_streaming.py`)](#non-streaming-mode-non_streamingpy)
  - [The `saw_partial_text` Guard — All Three Scenarios](#the-saw_partial_text-guard--all-three-scenarios)
  - [Runner Internals: Partial Events Are NOT Saved](#runner-internals-partial-events-are-not-saved)
- [Summary Table](#summary-table)
- [Running the Demos](#running-the-demos)
- [References](#references)

---

# Part 1: The ADK Runtime Foundation

## What Is the Runtime?

Think of the ADK Runtime as the **engine** of your agentic application. You define the parts (agents, tools), and the Runtime handles how they connect and run together to fulfill a user's request. It manages:

- Receiving user input
- Calling agents and LLMs
- Executing tools
- Committing state changes
- Streaming responses back to the UI

You never write the Runtime. You write **Agents**, and the Runtime executes them.

---

## The Event Loop: Core Mechanism

At its heart, the ADK Runtime operates on an **Event Loop** — a cooperative yield/pause/resume cycle between the **Runner** (orchestrator) and your **Execution Logic** (agents, tools, callbacks).

```
1. Runner receives user query
2. Runner appends it to session history via SessionService
3. Runner calls agent.run_async(context)
4. Agent processes, constructs an Event, yields it
5. Agent PAUSES immediately after yield
6. Runner receives Event, commits state/artifact changes via SessionService
7. Runner yields Event upstream to YOUR code (UI/caller)
8. Runner signals agent to RESUME
9. Agent resumes, sees updated state committed by Runner
10. Repeat until agent finishes (generator exhausted)
```

**This is the single most important pattern in ADK.** Every agent interaction — streaming or not — follows this exact loop. The `yield`/pause/resume cycle ensures state changes are committed before the agent continues, so the agent always sees the most up-to-date session state.

---

## Key Runtime Components

| Component | What It Is | Role |
|-----------|-----------|------|
| **Runner** | `google.adk.runners.Runner` | Orchestrator — drives the event loop, processes events, commits state |
| **InMemoryRunner** | `google.adk.runners.InMemoryRunner` | Convenience wrapper combining Runner + InMemorySessionService |
| **Agent** | `LlmAgent`, `BaseAgent`, etc. | Execution logic — thinks, decides, calls tools, yields events |
| **Event** | `google.adk.events.Event` | Atomic message carrying content + actions (state deltas, tool calls) |
| **RunConfig** | `google.adk.agents.RunConfig` | Runtime behavior settings — streaming mode, LLM call limits, speech |
| **SessionService** | `InMemorySessionService`, etc. | Persistence layer — stores session state and event history |
| **Session** | A data container | Holds `state` dict + list of past `events` for one conversation |
| **Invocation** | A concept | Everything from one user query to the agent finishing its response |

---

## Runner: The Orchestrator

The Runner is the central coordinator for a single user invocation. From the [ADK Event Loop docs](https://adk.dev/runtime/event-loop/#runners-role-orchestrator):

```python
# Simplified view of Runner's main loop (from official ADK docs)
async def run(new_query, ...) -> AsyncGenerator[Event, None]:

    # 1. Append user query to session history
    session_service.append_event(
        session,
        Event(author='user', content=new_query)
    )

    # 2. Kick off agent execution
    agent_event_generator = agent_to_run.run_async(context)

    # 3. Process each yielded event
    async for event in agent_event_generator:

        # 3a. Commit state/artifact deltas to session
        session_service.append_event(session, event)

        # 4. Yield event upstream (to UI/caller)
        yield event

        # 5. Agent resumes after this yield
        #    (implicit — the async generator continues)
```

**What the Runner does for EVERY event:**

1. **Receives** the event from agent's `yield`
2. **Commits** the event to session via `session_service.append_event()` — this applies `state_delta`, records `artifact_delta`, and appends to event history
3. **Yields** the event to YOUR code (your `async for event in runner.run_async(...)` loop)
4. **Waits** for you to process, then lets the agent resume

**What the Runner does NOT do:**
- It does NOT run agent logic — that's the agent's job
- It does NOT call the LLM — that's the agent's internal flow
- It does NOT execute tools — that's handled inside the agent's `_run_async_impl`

The Runner is purely the **coordinator** — it ensures state is committed and events flow correctly between the agent and your application.

### Runner vs InMemoryRunner

| Class | Setup | SessionStorage | Use Case |
|-------|-------|---------------|----------|
| `Runner` | Explicit: create `SessionService`, pass it in | Configurable (memory, database, etc.) | Production apps |
| `InMemoryRunner` | Implicit: creates `InMemorySessionService` internally | In-memory (lost on restart) | Demos, testing |

```python
# Runner: explicit setup (used in non_streaming.py)
session_service = InMemorySessionService()
await session_service.create_session(app_name=APP, user_id=USER, session_id=SID)
runner = Runner(session_service=session_service, app_name=APP, agent=root_agent)

# InMemoryRunner: convenience (used in main.py)
runner = InMemoryRunner(agent=root_agent, app_name=APP)
session = await runner.session_service.create_session(app_name=APP, user_id=USER)
```

Both produce the same event loop behavior. `InMemoryRunner` just bundles the setup.

---

## Event: The Message Protocol

Every communication in the event loop is an `Event`. From the [ADK source](https://github.com/google/adk-python/blob/main/src/google/adk/events/event.py):

```python
class Event(BaseEvent):
    # Who created this event
    author: str                          # "user", "agent_name", "model"

    # What content it carries
    content: Optional[types.Content]     # text, function calls, function responses

    # What side effects it causes
    actions: EventActions                # state_delta, artifact_delta, transfer_to_agent, escalate

    # Streaming flag
    partial: bool = False                # True = incremental chunk, False = final event

    # Control signals
    turn_complete: bool = False          # True = agent finished this turn
    interrupted: bool = False            # True = agent was interrupted

    # Metadata
    invocation_id: str                   # Links all events in one user query
    timestamp: float                     # When the event was created
    error_code: Optional[str]            # Error code if something went wrong
    error_message: Optional[str]         # Human-readable error
```

**Key fields for streaming:**

| Field | Streaming Relevance |
|-------|-------------------|
| `partial` | `True` = incremental chunk (not saved to session). `False` = final event (saved to session). |
| `content` | Carries the text/function_call/function_response data |
| `turn_complete` | `True` when the agent has finished responding to the user |
| `is_final_response()` | Method that checks if this event is the final response to display to the user |

---

## RunConfig: Controlling Runtime Behavior

`RunConfig` is passed to `runner.run_async()` to control how the agent behaves at runtime. From the [ADK Runtime Config docs](https://adk.dev/runtime/runconfig/):

```python
from google.adk.agents.run_config import RunConfig, StreamingMode

config = RunConfig(
    streaming_mode=StreamingMode.SSE,    # How responses are delivered
    max_llm_calls=200,                    # Safety cap on LLM calls per turn
    support_cfc=False,                    # Compositional Function Calling
)

async for event in runner.run_async(
    ...,
    run_config=config,
):
    ...
```

### `StreamingMode` — The Key Enum

From [ADK source: `run_config.py`](https://github.com/google/adk-python/blob/main/src/google/adk/agents/run_config.py):

```python
class StreamingMode(Enum):
    NONE = None       # Default. One complete response per turn. No partial events.
    SSE = 'sse'       # Server-Sent Events. Runner yields partial events as LLM generates.
    BIDI = 'bidi'     # Bidirectional. Reserved for runner.run_live() only.
```

| Mode | What happens inside ADK | When to use |
|------|------------------------|-------------|
| `StreamingMode.NONE` | `_call_llm_async` drops all `partial=True` events. Only the final aggregated event reaches your code. | CLI tools, batch processing, simple scripts |
| `StreamingMode.SSE` | `_call_llm_async` yields both partial AND non-partial events. You see chunks as they arrive. | Chat UIs, typewriter effects, web frontends |
| `StreamingMode.BIDI` | Not used in `run_async()`. Only for `run_live()` with WebSocket. | Real-time audio/video agents |

### Other RunConfig Fields

| Field | Purpose | Default |
|-------|---------|---------|
| `max_llm_calls` | Safety cap — stops execution if LLM is called too many times | 500 |
| `support_cfc` | Enable Compositional Function Calling (experimental, SSE only) | `False` |
| `speech_config` | Voice/language settings for audio agents | `None` |
| `response_modalities` | `["TEXT"]` or `["AUDIO", "TEXT"]` | `["TEXT"]` |
| `save_live_blob` | Save audio/video data to session for live agents | `False` |
| `get_session_config` | Limit events loaded per invocation (num_recent_events, after_timestamp) | Full history |
| `context_window_compression` | Compress context when approaching model limits | Disabled |

---

## SessionService & Session

The **SessionService** manages sessions. The **Session** holds the conversation state.

```python
# Creating a session (required before running the agent)
session_service = InMemorySessionService()
session = await session_service.create_session(
    app_name="my_app",      # Application identifier
    user_id="user_123",     # User identifier
    session_id="abc-123",   # Optional — auto-generated if omitted
)

# What a Session contains:
# session.state   — dict of key/value pairs (mutable by agent)
# session.events  — list of all Events in this conversation
```

**State commitment flow:**

```
Agent modifies state:    ctx.session.state['key'] = 'value'
         |
         v
Agent yields Event:      Event(actions=EventActions(state_delta={'key': 'value'}))
         |
         v
Runner processes:        session_service.append_event(session, event)
         |
         v
Session state updated:   session.state['key'] == 'value'  (guaranteed after yield resumes)
```

---

## A Full Invocation Walkthrough

Here's what happens when a user asks "What's the weather in London?" with a tool-calling agent:

```
STEP  WHAT HAPPENS                                              WHO DOES IT
────  ────────────────────────────────────────────────────────  ──────────
 1    User sends: "What's the weather in London?"               User
 2    Runner appends user Event to session history              Runner
 3    Runner calls agent.run_async(ctx)                         Runner
 4    Agent sends prompt to LLM                                 Agent
 5    LLM responds: function_call(get_weather, {city:"London"}) LLM
 6    Agent yields FunctionCall event                           Agent
 7    Agent PAUSES after yield                                  Agent
 8    Runner commits FunctionCall event to session              Runner
 9    Runner yields event upstream to YOUR code                 Runner
10    Agent RESUMES                                             Agent
11    Agent executes get_weather("London") -> {"temp": "15°C"}  Agent
12    Agent yields FunctionResponse event                      Agent
13    Agent PAUSES after yield                                  Agent
14    Runner commits FunctionResponse event to session          Runner
15    Agent RESUMES                                             Agent
16    Agent sends tool result back to LLM                       Agent
17    LLM responds: "The weather in London is cloudy, 15°C"     LLM
18    Agent yields final text event                             Agent
19    Agent PAUSES after yield                                  Agent
20    Runner commits final event to session                     Runner
21    Runner yields event upstream to YOUR code                 Runner
22    Agent RESUMES, generator exhausted                        Agent
23    Runner completes                                          Runner
```

**In streaming mode**, step 17 would emit multiple partial events as the LLM generates tokens.
**In non-streaming mode**, step 17 yields one complete event.

---

# Part 2: Streaming vs Non-Streaming

Now that you understand the runtime, here's what changes between the two modes.

## What `event.partial` Actually Means

Every Event has a boolean field called `partial`:

- **`partial=False`** (default): This is a "real" event. The Runner **saves it to session history** via `session_service.append_event()`. It represents a complete, finalized piece of content.
- **`partial=True`**: This is an incremental streaming chunk. The Runner does **NOT** save it to session history. It only forwards it to your code for real-time display.

```python
# From ADK Runner source (runners.py) — the critical filter:
async for event in agent_event_generator:
    if not event.partial:
        session_service.append_event(session, event)  # Only non-partial saved
    yield event  # ALL events go to your code
```

**The key rule: partial events are ephemeral streaming hints. Non-partial events are the ground truth.**

---

## The Two-Level Streaming Switch

Streaming requires **both** levels to be enabled:

```
Level 1: LiteLLM Model Config (HTTP layer)
─────────────────────────────────────────────
stream=True   → LiteLLM uses HTTP SSE streaming to call the LLM API
                LLM returns tokens incrementally over the wire

stream=False  → LiteLLM uses standard HTTP request
                LLM returns full response at once


Level 2: ADK RunConfig (event processing layer)
─────────────────────────────────────────────
StreamingMode.SSE  → ADK's _call_llm_async yields partial events
                     partial=True events reach your code

StreamingMode.NONE → ADK's _call_llm_async DROPS partial events
                     only partial=False events reach your code
```

```
                    stream=True          stream=False
                   ┌─────────────┐      ┌─────────────┐
                   │ LLM sends   │      │ LLM sends   │
                   │ chunks over │      │ full reply  │
                   │ HTTP SSE    │      │ at once     │
                   └──────┬──────┘      └──────┬──────┘
                          │                    │
              ┌───────────┼────────────────────┼───────────┐
              │           │   ADK _call_llm    │           │
              │           │   async filter     │           │
              │           ▼                    ▼           │
   Mode.SSE   │  yields partial=True    yields partial=True│
              │  + partial=False        + partial=False     │
              │           │                    │           │
   Mode.NONE  │  DROPS    │    DROPS partial  │           │
              │  partial  │    yields only     │           │
              │           │    partial=False   │           │
              └───────────┼────────────────────┼───────────┘
                          │                    │
                          ▼                    ▼
                    Your code receives events
```

---

## Streaming Mode (`main.py`)

```python
# Level 1: LiteLLM config — stream=True (line 41)
model = LiteLlm(
    model="openai/mercury-2",
    stream=True,                          # HTTP SSE streaming ON
    ...
)

# Level 2: RunConfig — StreamingMode.SSE (line 76)
async for event in runner.run_async(
    ...,
    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
):
    if event.partial:
        print(text, end='', flush=True)   # Display chunk immediately
    # Final aggregated event (partial=False) arrives after all chunks
```

**What you see:** Responses appear token-by-token (typewriter effect).

**What happens internally:**

```
Event 1:  partial=True,  text="The"           ← printed immediately
Event 2:  partial=True,  text=" answer"       ← printed immediately
Event 3:  partial=True,  text=" is"           ← printed immediately
Event 4:  partial=True,  text=" 42."          ← printed immediately
Event 5:  partial=False, text="The answer is 42."  ← SAVED to session, NOT printed (guard skips)
```

---

## Non-Streaming Mode (`non_streaming.py`)

```python
# Level 1: LiteLLM config — stream=False (line 29)
model = LiteLlm(
    model="openai/mercury-2",
    stream=False,                         # Standard HTTP request
    ...
)

# Level 2: No RunConfig — defaults to StreamingMode.NONE
async for event in runner.run_async(...):
    if event.is_final_response():
        final_response_text = event.content.parts[0].text
        break
```

**What you see:** Complete response appears all at once.

**What happens internally:**

```
Event 1:  partial=False, text="The answer is 42."  ← Only event, SAVED to session
```

ADK's `_call_llm_async` dropped all partial events. Only the final event reached your code.

---

## The `saw_partial_text` Guard — All Three Scenarios

This is the code in `main.py:69-93`:

```python
saw_text = False
saw_partial_text = False

async for event in runner.run_async(...):
    if not event.content:
        continue
    text = ''.join(part.text for part in event.content.parts if part.text)
    if not text:
        continue

    if event.partial:                        # ← Branch A: streaming chunk
        print(text, end='', flush=True)
        saw_text = True
        saw_partial_text = True
        continue

    # Branch B: final aggregated event (partial=False)
    if not saw_partial_text:                 # ← THE GUARD
        print(text, end='', flush=True)
        saw_text = True
```

The guard handles **three different scenarios** depending on what the LLM backend does:

### Scenario 1: Normal Streaming (Most Common)

LLM emits incremental partials, then a final aggregated event.

```
saw_partial_text = False  (start)

Event 1: partial=True,  text="The"
  → Branch A: print "The", saw_partial_text = True

Event 2: partial=True,  text=" answer is 42."
  → Branch A: print " answer is 42."

Event 3: partial=False, text="The answer is 42."  ← final aggregated
  → Branch B: if not saw_partial_text → False → SKIP (already printed)
```

Result: `The answer is 42.` — printed once, no duplicate. Correct.

**Without the guard:** Event 3 would print "The answer is 42." a second time. Double output.

### Scenario 2: No-Partial Backend (Critical Edge Case)

Some LiteLLM providers never emit `partial=True`. The full response arrives as a single `partial=False` event.

```
saw_partial_text = False  (start)

Event 1: partial=False, text="The answer is 42."  ← ONLY event
  → Branch B: if not saw_partial_text → True → PRINT
```

Result: `The answer is 42.` — printed from the final event. Correct.

**Without the guard (if Branch B didn't exist):** Nothing would print. Silent data loss.

**When this happens:**
- Provider falls back to non-streaming during outages
- Very short responses ("Yes", "OK") arrive in one chunk
- Network proxies buffer SSE chunks and deliver them all at once
- Some reasoning models buffer their entire thinking process

### Scenario 3: No Text Response

Agent returns events with content but no text parts.

```
saw_partial_text = False, saw_text = False  (start)

Event 1: partial=False, text=""  ← empty
  → if not text: continue  ← skipped

Loop ends. saw_text is still False.
  → print "(no text response)"
```

Result: `(no text response)` — user sees that nothing came back. Correct.

### The Decision Tree

```
Event arrives
    │
    ├── event.content is None?  ──────→  skip
    │
    ├── text is empty?  ───────────→  skip
    │
    ├── event.partial is True?  ───→  print chunk (typewriter)
    │       │                         saw_partial_text = True
    │       └─────────────────────→  continue to next event
    │
    └── event.partial is False?  ──→  FINAL aggregated event
            │
            ├── saw_partial_text=True?   → SKIP (already printed chunks)
            └── saw_partial_text=False?  → PRINT (only text we'll get)
```

---

## Runner Internals: Partial Events Are NOT Saved

From the ADK Runner source code ([`runners.py`](https://github.com/google/adk-python/blob/main/src/google/adk/runners/runner.py)):

```python
async for event in agent_event_generator:
    # Only non-partial events are committed to session history
    if not event.partial:
        session_service.append_event(session, event)

    # ALL events (partial and non-partial) go to your code
    yield event
```

This is intentional:
- **Partial events are ephemeral** — they are streaming hints, not conversation history
- **Only the final aggregated event is the ground truth** — it gets saved to session state
- **Session state stays clean** — no fragmented text chunks in your history
- **Agent resumption works correctly** — when the agent resumes after yield, it sees the complete committed state, not partial fragments

---

## Summary Table

| Property | Streaming (`main.py`) | Non-Streaming (`non_streaming.py`) |
|----------|----------------------|-----------------------------------|
| `LiteLlm.stream` | `True` | `False` |
| `RunConfig.streaming_mode` | `StreamingMode.SSE` | `StreamingMode.NONE` (default) |
| `_call_llm_async` yields partials? | Yes | No (filtered out) |
| `event.partial` values seen | `True` then `False` | Always `False` |
| Events saved to session | Only `partial=False` | The single response event |
| How to detect completion | Check `event.partial == False` | Check `event.is_final_response()` |
| Text in partial events | Only new tokens since last chunk | N/A |
| Text in final event | Full merged text (ADK concatenated) | Complete response |
| Runner class | `InMemoryRunner` | `Runner` + `InMemorySessionService` |
| `input()` handling | `loop.run_in_executor()` (non-blocking) | Direct `input()` (blocks event loop) |
| Typical use case | Chat UIs, web frontends, CLI typewriter | Batch processing, simple scripts |

---

## Running the Demos

### Prerequisites

```bash
pip install google-adk litellm python-dotenv icecream
echo "MERCURY_API_KEY=your_key_here" > .env
```

### Streaming

```bash
cd streaming
python main.py
```

Responses appear token-by-token. Type `exit` to quit.

### Non-Streaming

```bash
cd streaming
python non_streaming.py
```

Complete response appears all at once. Type `exit` to quit.

---

## References

- [ADK Event Loop Docs](https://adk.dev/runtime/event-loop/) — Runner's role, yield/pause/resume cycle, key components
- [ADK Runtime Config](https://adk.dev/runtime/runconfig/) — RunConfig, StreamingMode enum, all config fields
- [ADK Streaming Guide](https://adk.dev/streaming/dev-guide/part4/) — SSE vs BIDI, progressive streaming
- [ADK Agent Team Tutorial](https://adk.dev/tutorials/agent-team/) — Multi-agent delegation, LiteLLM integration
- [GitHub Discussion #4725](https://github.com/google/adk-python/discussions/4725) — event.partial behavior explained
- [ADK Source: `base_llm_flow.py`](https://github.com/google/adk-python/blob/main/src/google/adk/flows/llm_flows/base_llm_flow.py) — Where partials are yielded or dropped
- [ADK Source: `runner.py`](https://github.com/google/adk-python/blob/main/src/google/adk/runners/runner.py) — Where partials skip session save
- [ADK Source: `run_config.py`](https://github.com/google/adk-python/blob/main/src/google/adk/agents/run_config.py) — StreamingMode definitions
- [ADK Source: `event.py`](https://github.com/google/adk-python/blob/main/src/google/adk/events/event.py) — Event fields including partial
