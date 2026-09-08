# State vs Cache: Understanding Long-Horizon Evolving State

**Claim:** A fixed-shape recurrent state can process a sequence of unbounded duration without allocating a new memory slot for every token, but information stored in that fixed state can interfere, causing retrieval failures.

**Claim status:** This claim is demonstrated live, in-browser, on a small toy retrieval task. It is not a benchmark result and does not reproduce any published BDH numbers, training run, or evaluation.

**Live demo:** https://state-vs-cache.vercel.app/

**Repository:** https://github.com/aryanharit14-ux/state-vs-cache

---

## Audience and prerequisites

**Intended learner:** Students, developers, and early-career ML practitioners
who understand basic vectors and matrix multiplication and want an intuitive,
interactive introduction to fixed-state sequence memory. No prior knowledge
of BDH is required.

**Prerequisites:** Basic familiarity with vectors, dot products, and matrix
multiplication. A rough understanding of the idea that a Transformer can
retain past information in a growing KV cache is helpful, but no prior
knowledge of BDH is assumed.

**Learning objectives:** After using this explainer, the learner should be
able to:
- explain why a fixed-size recurrent state can process sequences without
  allocating a new memory slot for every token;
- distinguish a growing exact cache from a fixed-size recurrent state;
- explain how interference between stored information can cause retrieval
  failures in a fixed-capacity state; and
- connect the toy's additive outer-product write/read mechanism to the
  idealized BDH formulation while recognizing the limits of that analogy.

## What this demonstrates

Two memory systems are given the same stream of randomly generated key-to-value facts, and both are queried on every stored key:

- A **growing cache** — a plain dictionary that allocates one slot per fact. Exact recall, but memory grows linearly with the number of facts.
- A **fixed recurrent state** — a single D×D matrix that all facts are written into via additive outer-product updates, inspired by the idealized synaptic memory update in Pathway's Dragon Hatchling (BDH) architecture. Memory size never grows, but retrieval can degrade as more facts are superimposed into the same fixed-size structure.

The learner controls two real variables — the number of stored facts (N) and the state's dimensionality (D) — and watches retrieval accuracy, a live sigma matrix heatmap, and per-key query results respond to real, in-browser computation.

## The experiment

### Growing Cache

- Implementation: a JavaScript `Map`, one entry per fact.
- Retrieval: exact key lookup. No interference is possible by construction — this is verified by the actual computation, not asserted.

### Fixed Recurrent State

- Implementation: a `D×D` matrix `sigma`, initialized to zero.
- Write: for each fact, `sigma += outer(key_vector, value_vector)`.
- Read: for a query key, compute `output = key_vector @ sigma`, then find the nearest stored value vector by cosine similarity.
- Interference: as more facts are written into the same fixed matrix, non-orthogonal key vectors begin to overlap, and queries can retrieve a blended, incorrect result.

## Interact with the experiment

Open the live demo. It loads pre-configured at **N=18, D=8**, where the fixed recurrent state is already showing retrieval failures while the cache remains exact. Move the **Number of facts** and **State dimension** sliders to explore the trade-off directly. Use the **Query Inspector** to check any individual key against both systems' actual output and the ground truth.

## How the computation works

Every result shown — accuracy percentages, the sigma heatmap, and individual query outcomes — is computed live in the browser on every slider change. Nothing is precomputed or scripted; there is no cached or hardcoded accuracy value anywhere in the implementation.

### Cache

`cacheCorrect = (cache.get(key) === fact.value)`, aggregated across all N queries.

### Recurrent state

`sigma` is freshly allocated and rebuilt from scratch on every simulation run. Read accuracy is computed by comparing each query's nearest-neighbor prediction against the fact's actual stored value.

## Experiment note — result variance and reproducibility

Because key and value vectors are randomly generated, individual random seeds can produce different retrieval accuracy at the same N and D. The demo uses a fixed seed (`mulberry32`, seed `0xDEADBEEF`), so its results are deterministic and fully reproducible on every load and reload.

To check whether this seed's result was representative rather than an unusually convenient outlier, a 25-seed sensitivity check was run on the production configuration (N=18, D=8) without modifying the deployed app.

| Metric | Value |
|---|---:|
| Production seed result | 50.0% |
| Mean across 25 seeds | 42.7% |
| Median | 44.4% |
| Range | 22.2% to 66.7% |
| Seeds within 15 percentage points of 50% | 18 / 25 |

The production result falls within the typical range observed across independent seeds — it is not a cherry-picked outlier.

The same sensitivity check was run at three other configurations to confirm the qualitative trend holds beyond a single seed:

| N | D | Mean accuracy | Median | Range | Mean failures |
|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 97.0% | 100% | 75.0% to 100% | 0.12 |
| 18 | 8 | 42.7% | 44.4% | 22.2% to 66.7% | 10.32 |
| 30 | 8 | 20.7% | 20.0% | 6.7% to 36.7% | 23.80 |
| 18 | 24 | 98.2% | 100% | 88.9% to 100% | 0.32 |

In this toy experiment, increasing the number of stored facts while holding the state dimension fixed produces progressively more retrieval failures across this 25-seed sensitivity sample. Increasing the state dimension at fixed load reduces observed interference. Individual seeds are not perfectly monotonic — interference is a property of how orthogonal a particular set of randomly drawn key vectors happens to be, not a fixed deterministic curve — but the aggregate trend across this 25-seed sensitivity sample is consistent. Cache accuracy remained 100% across every configuration and every seed tested, as expected from exact lookup.

## Connection to Dragon Hatchling (BDH)

The write/read rule used in this toy shares its mathematical structure with the idealized formulation described in Pathway's "From attention to synapses: deriving BDH" (Chapter 2, BDH Explainer Series):

```text
sigma_t = sigma_(t-1) + x_t^T v_t
o_t = x_t · sigma_t
```

This toy experiment isolates that specific associative matrix memory update to illustrate how fixed-size state capacity limits and interference occur in practice.

## Scope of this analogy

Our toy's write/read rule — `sigma += outer(key, value)`, `read = key @ sigma` — has the same additive outer-product write/read structure as the idealized BDH formulation shown above.

It is **not** a reproduction of the trained BDH architecture. Our toy uses a small engineered key/value retrieval task, while BDH includes additional learned dynamics and nonlinearities around this mechanism.

Pathway's architecture comparison, paraphrased rather than quoted, describes BDH's runtime memory as a fixed-size synaptic state whose effective context length is bounded by information capacity rather than sequence length. This contrasts with a Transformer's key-value cache, which grows as new tokens are added.

**Related:** BDH-CQ (Pathway, Aug 2026) is a related system in the same family that adapts via recurrent state at inference time without updating its parameters. This is a secondary contextual reference; the toy's central mechanism is the additive outer-product write, not BDH-CQ's task-adaptation procedure.

## What this toy does NOT reproduce

- BDH's ReLU nonlinearity and enforced sparse, non-negative activity.
- BDH's transition operator `U`, which handles positional and decay effects; this toy has no temporal ordering beyond write sequence.
- BDH's low-rank factorization (`E`, `Dx`, `Dy`), used for GPU efficiency; this toy uses a full dense `D×D` matrix.
- Any of BDH's trained parameters, training procedure, or published benchmark results, including ARC-AGI-1 or long-context evaluations. This toy illustrates one isolated mechanism, not BDH's behavior as a trained model.

## Evidence and references

Primary sources used to verify the technical claims in this project:

1. Pathway, *The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain* (arXiv 2509.26507, 2025) — source of the core synaptic state update/read equations.
2. Pathway, *Reasoning at a Fraction of the Compute* and the accompanying BDH-CQ technical report (Aug 2026) — source for the BDH-CQ contextual reference.
3. *A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets* (arXiv 2607.02303, 2026) — directly relevant work on interference/forgetting in fixed-size recurrent memory.
4. *LoLA: Low-Rank Linear Attention With Sparse Caching* (arXiv 2505.23666, 2025) — describes memory collision from non-orthogonal keys in linear-attention-style memory, closely related to this project's failure mode.

Do not invent URLs, DOI numbers, author names, benchmark numbers, or additional references.

## Architecture

Single self-contained `index.html` file — no build step, no dependencies, no server component.

- **Simulation:** vanilla JavaScript, `runSim(N, D)` — generates facts, builds the cache, builds and reads the sigma matrix, and computes accuracy.
- **Visualization:** HTML canvas for the sigma heatmap; DOM/CSS for accuracy bars, the ledger table, and the panel layout.
- **Interaction:** native HTML sliders and a `<select>` for the Query Inspector; a native `<details>` element for the collapsible explanation panel.

Everything shown is computed live in the browser on every input change. Nothing in this project is precomputed, cached, or hardcoded.

## Reproducibility

No installation is required. Open `index.html` in any modern browser, or visit the live demo link above.

To verify the deterministic seed behavior described above, reload the page or move a slider back to N=18, D=8. The result will be identical every time:

- State accuracy: 50.0%
- Cache accuracy: 100%

## AI assistance disclosure

- **Antigravity** (Google's agentic development environment) generated the HTML/CSS/JS implementation from a detailed specification, including the exact write/read equations, toy task design, UI requirements, and a subsequent visual redesign pass.
- **Independently verified by the author:** the BDH equations against the primary source (Pathway, *From attention to synapses: deriving BDH*, Chapter 2); the four cited papers and their relevance; the live experiment's numerical behavior through manual multi-configuration testing and a 25-seed sensitivity check; and the implementation's integrity, including live recomputation, absence of hardcoded accuracy values, and dynamic Query Inspector behavior.
- **The author can defend:** the interference mechanism and why it occurs mathematically, why the core claim is falsifiable and how the demo tests it, what the toy deliberately omits from BDH's real architecture, the four cited papers and their relevance, and the distinction between this toy's empirical sensitivity check and a formal statistical or mathematical proof.

## License

MIT License

Copyright (c) 2026 Aryan Harit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Credits

Built for DataForge 2026 (Pathway Track), IIT Kharagpur, in partnership with Pathway.

Not affiliated with or endorsed by Pathway. No BDH benchmark results are reproduced anywhere in this project.
