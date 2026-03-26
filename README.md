# neuram

Memory for AI agents that works the way your brain does. Not a vector database with a fancy name — actual neuroscience, implemented.

`neuram` (neural + RAM) uses real brain regions as first-class Python objects. Your agent has a Hippocampus. It has a PrefrontalCortex that holds 7 things at once, just like yours. Memories decay when ignored and survive when revisited. At idle, a sleep cycle quietly consolidates what matters and prunes what doesn't.

If you've ever built an agent and wondered why it keeps "forgetting" things it should know, or why it treats a one-off fact the same as something the user says every single day — this is the missing piece.

---

## How it works

```
 ╔══════════════════════════════════════════════════════════════════╗
 ║                     PERCEPTION PIPELINE                          ║
 ╠══════════════════════════════════════════════════════════════════╣
 ║                                                                  ║
 ║  raw input                                                       ║
 ║      │                                                           ║
 ║      ▼                                                           ║
 ║  ┌─────────────────┐                                            ║
 ║  │  SensoryCortex  │  holds raw input for 3s, then gone         ║
 ║  └────────┬────────┘                                            ║
 ║           │                                                      ║
 ║           ▼                                                      ║
 ║  ┌─────────────────┐                                            ║
 ║  │    Thalamus     │  salience too low? ──────────── ✗ dropped  ║
 ║  └────────┬────────┘                                            ║
 ║           │ passed attention gate                                ║
 ║           ▼                                                      ║
 ║  ┌─────────────────┐                                            ║
 ║  │    Amygdala     │  tags emotional weight, boosts stability   ║
 ║  └────────┬────────┘                                            ║
 ║           │                                                      ║
 ║           ▼                                                      ║
 ║  ┌─────────────────────────────────────────────────────────┐   ║
 ║  │               PrefrontalCortex                          │   ║
 ║  │  slot 1 │ slot 2 │ slot 3 │ slot 4 │ slot 5 │ ...      │   ║
 ║  │         7 slots max · LRU eviction · decays over time   │   ║
 ║  └────────────────────────┬────────────────────────────────┘   ║
 ║                           │ indexed + temporally linked         ║
 ║                           ▼                                     ║
 ║  ┌─────────────────────────────────────────────────────────┐   ║
 ║  │                    Hippocampus                          │   ║
 ║  │   pattern completion · spreading activation · recall    │   ║
 ║  └────────────────────────┬────────────────────────────────┘   ║
 ║                           │                                     ║
 ╚═══════════════════════════╪═════════════════════════════════════╝
                             │
                   sleep cycle runs here
                   (background · consolidate → decay → prune)
                             │
 ╔═══════════════════════════╪═════════════════════════════════════╗
 ║           LONG-TERM MEMORY  (Redis)                             ║
 ╠═══════════════════════════╪═════════════════════════════════════╣
 ║              ┌────────────┼────────────┐                        ║
 ║              ▼            ▼            ▼                        ║
 ║       ┌──────────┐ ┌────────────┐ ┌─────────────┐             ║
 ║       │Neocortex │ │ Cerebellum │ │ BasalGanglia│             ║
 ║       │          │ │            │ │             │              ║
 ║       │ semantic │ │ procedural │ │  habitual   │              ║
 ║       │ episodic │ │   skills   │ │  + reward   │              ║
 ║       └──────────┘ └────────────┘ └─────────────┘             ║
 ║                                                                  ║
 ║   Ebbinghaus decay runs on all LTM. Below 5% → pruned.         ║
 ╚══════════════════════════════════════════════════════════════════╝
```

Everything lives in-memory until it earns a spot in Redis. The same way your brain doesn't bother encoding every conversation into long-term storage — only the things that matter make it through.

| Region | What it does | Where |
|---|---|---|
| SensoryCortex | Raw input buffer. Expires in 3 seconds if nobody pays attention. | In-memory |
| Thalamus | The bouncer. Filters out low-relevance input before it reaches working memory. | In-memory |
| Amygdala | Tags emotional weight. High-salience memories get a stability boost right at encoding. | In-memory |
| PrefrontalCortex | Working memory. 7±2 slots, LRU eviction, decays without rehearsal. | In-memory |
| Hippocampus | Episodic index. Handles pattern completion and temporal association. | In-memory |
| Neocortex | Long-term semantic and episodic storage. | Redis |
| Cerebellum | Procedural memory — learned skills and workflows. Very hard to unlearn once consolidated. | Redis |
| BasalGanglia | Habit memory, modulated by a dopamine-style reward signal. | Redis |
| SleepCycle | Background consolidation. Replays working memory, promotes to LTM, prunes the rest. | — |

---

## The memory unit: Engram

Richard Semon coined "engram" in 1904 to describe the physical trace a memory leaves in the brain. That's what we store.

```python
@dataclass
class Engram:
    content: str
    memory_type: MemoryType     # EPISODIC | SEMANTIC | PROCEDURAL | HABITUAL
    salience: float             # 0–1, set by the Amygdala
    embedding: list[float]      # sentence-transformer vector
    synaptic_strength: float    # current activation — decays via Ebbinghaus
    stability: float            # resistance to forgetting — grows with each recall
    access_count: int
    associations: list[tuple]   # spreading activation graph edges
    fragments: list[str]        # sub-engram IDs — memory is reconstructive, not holistic
```

Two properties matter most: `synaptic_strength` (how vivid the memory is right now) and `stability` (how long it'll last before fading). Stability grows every time the memory is recalled — that's Long-Term Potentiation.

---

## The biology, briefly

**Forgetting curve.** Ebbinghaus figured this out in 1885 by memorizing nonsense syllables and testing himself obsessively. The decay follows:

```
R(t) = e^(-t / S)
```

`t` is hours since last access. `S` is stability — which grows with each recall. So the more often you retrieve something, the longer it survives between retrievals. Spaced repetition falls out naturally from the math.

**Long-Term Potentiation.** Every recall event strengthens the engram:

```
stability += 0.3 × salience_boost × log(1 + access_count)
```

High-salience memories — the ones the Amygdala flagged as emotionally significant — grow faster. That's the norepinephrine effect.

**Spreading activation.** When you recall something, you don't just retrieve that one thing — you light up a neighborhood. A memory of "Paris" activates "France", "Eiffel Tower", "that trip in 2019". neuram does the same: cosine similarity gets initial activation, then it propagates through the association graph with a 0.6x decay per hop.

**Sleep cycle.** This is the one people don't build. While your agent is idle, a background coroutine runs: it replays working memory, promotes anything that earned it to long-term storage, applies Ebbinghaus decay to everything in Redis, and deletes engrams that fell below 5% retention. Tononi and Cirelli called this synaptic homeostasis — the brain literally downscales weak connections every night so the strong ones stand out.

---

## Installation

```bash
pip install neuram
```

You'll need Redis. If you don't have it: `brew install redis && redis-server`.

Dev dependencies (pytest, fakeredis):

```bash
pip install "neuram[dev]"
```

---

## Quick start

```python
import asyncio
from neuram import Brain, MemoryType

async def main():
    brain = await Brain.create(redis_url="redis://localhost:6379")

    # facts go to Neocortex after consolidation
    await brain.perceive("Paris is the capital of France",
                         memory_type=MemoryType.SEMANTIC)

    # high salience → Amygdala gives it a stability boost at encoding
    await brain.perceive("I had a great meeting with Alice today",
                         memory_type=MemoryType.EPISODIC,
                         salience=0.8)

    # skills end up in Cerebellum — hard to forget once learned
    await brain.perceive("To debug Python: add breakpoint(), run, inspect locals",
                         memory_type=MemoryType.PROCEDURAL)

    # reward signal goes to BasalGanglia (dopamine-style)
    await brain.perceive("User always prefers concise answers",
                         memory_type=MemoryType.HABITUAL,
                         reward=0.9)

    # recall via hippocampal pattern completion + spreading activation
    results = await brain.recall("capital city Europe")
    for engram, score in results:
        print(f"{score:.3f}  [{engram.memory_type}]  {engram.content}")

    print(f"Working memory: {len(brain.working_memory)} / 7 slots")

    # run a manual sleep cycle — or let start_sleep_cycle() do it in the background
    stats = await brain.sleep()
    print(stats)  # {'consolidated': 2, 'pruned_wm': 0, 'pruned_ltm': 1}

    await brain.close()

asyncio.run(main())
```

---

## A few things worth knowing

**Background sleep.** You probably want this running continuously:

```python
brain = await Brain.create(redis_url="redis://localhost", sleep_interval_seconds=30.0)
await brain.start_sleep_cycle()
# your agent does its thing
await brain.stop_sleep_cycle()
```

**Explicit forgetting.** Sometimes you need to surgically remove something:

```python
await brain.forget(engram.engram_id)
```

**Watching memories decay.** Useful for debugging or for building rehearsal schedules:

```python
from neuram import compute_retention, time_until_forgotten

compute_retention(engram)       # 0.73 — still pretty strong
time_until_forgotten(engram)    # 4.2 hours left before it drops below threshold
```

**Bypassing the Brain and working directly with regions:**

```python
from neuram import Hippocampus

hpc = Hippocampus()
results = hpc.pattern_complete("what did we discuss about Redis?", top_k=10)
```

---

## Why not just use a vector DB?

You can. pgvector, Chroma, Pinecone — they're all fine for search. But they don't forget. They don't have working memory limits. They treat a user's offhand comment the same as something they've said a hundred times. They don't do anything while your agent sleeps.

neuram is for when you want memory that behaves, not just memory that stores.

---

## References

- Semon, R. (1904). *Die Mneme* — coined "engram"
- Ebbinghaus, H. (1885). *Über das Gedächtnis* — forgetting curve
- Collins & Loftus (1975). Spreading activation in semantic networks
- Bliss & Lømo (1973). Long-term potentiation in hippocampus
- Tononi & Cirelli (2006). Synaptic homeostasis hypothesis
- Scoville & Milner (1957). Patient H.M. — hippocampus and declarative memory
- McCormick & Thompson (1984). Cerebellum and procedural memory

---

MIT License
