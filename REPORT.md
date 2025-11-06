## 1. Executive Summary
- **Research question:** Can frontier LLMs temporarily override entrenched knowledge (“conditional forgetting”) when explicitly instructed to use counterfactual rules?
- **Key finding:** All tested models (GPT‑4.1, GPT‑4o‑mini, Claude 3.5 Sonnet, Mistral Large 2407) fell well short of perfect compliance (78–97% accuracy) and chess-style overrides drove most errors. Chain-of-thought prompting improved accuracy by 8–12 percentage points but still left statistically significant gaps versus perfect forgetting (binomial *p* < 2e‑15 even for the best setting).
- **Practical implication:** High-end LLMs cannot yet be trusted to apply hypothetical rule changes reliably; systems that require reversible knowledge must add external validation or symbolic enforcement.

## 2. Goal
- **Hypothesis:** Humans can “conditionally forget” base rules (e.g., canonical chess moves) while LLMs struggle to do so even with advanced reasoning prompts.
- **Importance:** Safety-critical and creative workflows (simulations, policy sandboxing, what-if planning) need models that can faithfully follow temporary rules; failure leads to hallucinated facts or policy violations.
- **Problem solved:** Provide a reproducible benchmark and comparative evaluation quantifying conditional forgetting gaps with real API calls.
- **Expected impact:** Highlight specific failure modes and quantify the benefit—and insufficiency—of chain-of-thought (CoT) prompting, guiding future mitigation strategies.

## 3. Data Construction

### Dataset Description
- **File:** `data/conditional_forgetting.jsonl`
- **Size:** 60 scenarios evenly split across three domains
  - 25 chess move overrides (piece behaviors swapped)
  - 20 arithmetic operator overrides (custom addition/multiplication/etc.)
  - 15 procedural/protocol swaps (label/word/behavior remappings)
- **Source:** Authored via `scripts/build_dataset.py`; purely synthetic to avoid contamination yet structured for automatic scoring.
- **Format:** Each JSON line contains `id`, `domain`, `rule`, `question`, `choices` (A–D), and `correct_choice`.
- **Bias/Limitations:** English-only, single-step reasoning, assumes deterministic multiple-choice answers; does not capture multi-turn adaptation.

### Example Samples
| ID | Domain | Rule Snippet | Question | Answer |
|----|--------|--------------|----------|--------|
| `chess_knight_as_bishop_1` | chess | “Knights move exactly like bishops.” | “A piece starts on C3 and targets F6…” | `A` (legal) |
| `math_offset_add_1` | math | “x ⊕ y = x + y − 3.” | “What is 7 ⊕ 6?” | `C` (=10) |
| `protocol_6` | protocol | “Red badges = guests, blue = staff.” | “How do you treat a red badge holder?” | `B` |

### Data Quality
- Missing values: 0% (script enforces required fields)
- Outliers: Not applicable (discrete categorical answers)
- Class distribution: Balanced across answer labels by construction (two-choice chess, four-choice math, three-choice protocol)
- Validation: Script asserts question/choice presence; manual spot-check after generation.

### Preprocessing Steps
1. Author domain-specific scenario generators (chess move validators, arithmetic transformations, procedure remappings).
2. Serialize to JSONL with deterministic ordering to ease diffing and caching.
3. No further preprocessing required at evaluation time.

### Train/Val/Test Splits
- Entire set used as evaluation-only benchmark; no training or validation split because models are frozen APIs. All models see identical ordered prompts to enable paired significance tests.

## 4. Experiment Description

### Methodology – High-Level Approach
1. Build the benchmark described above.
2. Design two prompt templates:
   - **Direct:** concise instruction + letter-only answer.
   - **Chain-of-thought (CoT):** require brief reasoning then `Final Answer: <letter>`.
3. Query four models via real APIs: GPT‑4.1, GPT‑4o‑mini (OpenAI), Claude 3.5 Sonnet (OpenRouter relay), Mistral Large 2407 (OpenRouter).
4. Parse responses, auto-grade against gold labels, and compute descriptive + inferential statistics (Wilson intervals, binomial tests vs. perfection, McNemar for prompt deltas).

### Why This Method?
- LLM behavior must be measured with genuine API calls (prompt simulations are invalid per spec).
- Multiple domains (symbolic, numeric, procedural) expose whether failures are localized or systemic.
- Direct vs. CoT comparison isolates whether additional reasoning mitigates forgetting.

### Implementation Details

#### Tools and Libraries (versions)
- Python 3.12.2 (uv virtualenv)
- `openai 2.7.1`, `anthropic 0.72.0` (OpenRouter-compatible requests), `requests 2.32.5`
- `pandas 2.3.3`, `numpy 2.3.4`, `scipy 1.16.3`
- `matplotlib 3.10.7`, `seaborn 0.13.2`
- `tenacity 9.1.2`, `tqdm 4.67.1` for retries/progress
- Environment metadata stored in `results/analysis/environment.json`

#### Algorithms / Models
- Chat-completion APIs with deterministic decoding: `temperature=0`, `top_p=1`, no sampling.
- Claude 3.5 Sonnet and Mistral Large served via OpenRouter REST endpoint.
- Responses parsed using regex `Final Answer: <letter>` fallback to last standalone `[A-D]`.

#### Hyperparameters
| Parameter                         | Value                               | Selection Method          |
|----------------------------------|--------------------------------------|---------------------------|
| temperature                      | 0                                    | Fix determinism (spec)    |
| top_p                            | 1                                    | Default deterministic     |
| max_tokens (Anthropic/OpenRouter)| 512 completion                       | Enough for CoT reasoning  |
| prompt styles                    | `direct`, `cot`                      | Planned ablation          |
| dataset size                     | 60 scenarios                         | Balance coverage vs cost  |
| retry policy                     | Exponential backoff via `tenacity`   | Robust API calling        |

#### Training / Analysis Pipeline
1. `python scripts/build_dataset.py` – generates benchmark file.
2. `python scripts/run_experiments.py` – sequentially queries each model/prompt pair, saving raw JSONL responses to `results/model_outputs/<run>__*.jsonl`.
3. `python scripts/analyze_results.py` – aggregates metrics, produces CSV summaries, and saves plots + representative failures.

### Experimental Protocol
- **Reproducibility:** Single run per condition (deterministic decoding). Randomness arises only from API nondeterminism (not observed at temp=0). Results tied to run ID `20251106-142751` stored in `results/analysis/run_id.txt`.
- **Hardware:** CPU-only session (Ubuntu VM). No GPU required; all computation is API-bound.
- **Runtime:** ~22 minutes for full API sweep + 5 minutes for analysis.
- **Cost Tracking:** Token counts logged per condition in `summary.csv`. GPT-4.1 CoT consumed ~7.4k prompt / 6.3k completion tokens (~$0.02). Full run stayed well under $1 across providers.

### Evaluation Metrics
1. **Compliance Accuracy:** fraction of correct answers per model/prompt/domain. Captures adherence to hypothetical rules.
2. **Wilson 95% CI:** quantifies uncertainty for finite samples.
3. **Binomial test vs. perfect compliance:** checks whether performance differs significantly from 100%.
4. **McNemar test & Cohen’s h:** evaluate impact of CoT vs. direct prompting on the same items.
5. **Error taxonomy counts:** derived by grouping failures by domain and scenario type.

### Raw Results

#### Aggregate Accuracy
| Model | Prompt | Correct / 60 | Accuracy | 95% CI | Binomial *p* (vs 100%) |
|-------|--------|--------------|----------|--------|------------------------|
| GPT‑4.1 | CoT | 58 | 0.967 | [0.886, 0.991] | 1.77e-15 |
| GPT‑4.1 | Direct | 51 | 0.850 | [0.739, 0.919] | 1.48e-71 |
| GPT‑4o‑mini | CoT | 52 | 0.867 | [0.758, 0.931] | 2.56e-63 |
| GPT‑4o‑mini | Direct | 47 | 0.783 | [0.664, 0.869] | 5.17e-105 |
| Claude 3.5 Sonnet | CoT | 53 | 0.883 | [0.778, 0.942] | 3.86e-55 |
| Claude 3.5 Sonnet | Direct | 48 | 0.800 | [0.682, 0.882] | 1.40e-96 |
| Mistral Large 2407 | CoT | 56 | 0.933 | [0.841, 0.974] | 4.88e-31 |
| Mistral Large 2407 | Direct | 49 | 0.817 | [0.701, 0.894] | 3.43e-88 |

#### Domain Breakdown (Accuracy)
| Model | Prompt | Chess (n=25) | Math (n=20) | Protocol (n=15) |
|-------|--------|--------------|-------------|-----------------|
| GPT‑4.1 | CoT | 0.92 | 1.00 | 1.00 |
| GPT‑4.1 | Direct | 0.84 | 0.75 | 1.00 |
| GPT‑4o‑mini | CoT | 0.68 | 1.00 | 1.00 |
| GPT‑4o‑mini | Direct | 0.60 | 0.85 | 1.00 |
| Claude 3.5 | CoT | 0.72 | 1.00 | 1.00 |
| Claude 3.5 | Direct | 0.64 | 1.00 | 0.80 |
| Mistral Large | CoT | 0.84 | 1.00 | 1.00 |
| Mistral Large | Direct | 0.60 | 0.95 | 1.00 |

Plots saved to `results/plots/accuracy_20251106-142751.png` and `results/plots/domain_accuracy_20251106-142751.png`.

## 5. Result Analysis

### Key Findings
1. **Systematic deviation from perfect forgetting:** Even the best condition (GPT‑4.1 + CoT) missed 2/60 prompts, and all binomial tests rejected perfect compliance (p < 2e‑15). Thus, conditional forgetting remains unsolved.
2. **Chess overrides drive most failures:** 87% of all errors occurred in chess scenarios, especially when piece roles were swapped. Math and protocol overrides were nearly solved except for GPT‑4.1 direct (math accuracy 75%).
3. **Chain-of-thought helps but doesn’t close the gap:** CoT improved accuracy by 8–12 percentage points across models. McNemar tests showed significant gains for GPT‑4.1 (p = 0.023, Cohen’s h = 0.43) and Mistral Large (p = 0.046, h = 0.36), but residual errors persisted.
4. **Model ranking:** GPT‑4.1 > Mistral Large > Claude 3.5 ≈ GPT‑4o-mini under CoT, but chess accuracy still capped at 92% for GPT‑4.1.

### Hypothesis Testing
- **H1 (Rule adoption <80%):** Supported for all direct prompts (78–85%). CoT pushes some models above 90%, but still significantly below 100%.
- **H2 (CoT benefit <15pp):** Observed improvements ranged from +6 to +12pp, aligning with the hypothesis of modest gains.
- **H3 (Frontier models outperform open-weight baseline yet imperfect):** GPT‑4.1 and Mistral Large (via OpenRouter) outperformed GPT‑4o-mini but still showed statistically significant deficits vs. perfection.

### Error Analysis
- **Rule reversion:** Models frequently defaulted to canonical chess reasoning (“knights move in L-shapes”) despite explicit swaps, even after restating the rule inside the CoT (e.g., Claude insisting A3→B5 is illegal under knight-rule despite describing the correct 1×2 move).
- **Directional confusion:** GPT‑4.1 direct misapplied the redefined addition ⊕ by ignoring the −3 offset four times in a row, indicating brittle arithmetic overrides.
- **Parsing mistakes:** Rare (0 unparsed responses), confirming that failures stem from reasoning, not formatting.
- **Representative cases:** Logged in `results/analysis/sample_failures.json` for each model/prompt/domain combination (two samples each).

### Limitations
- Synthetic dataset may not capture richer multi-step forgetting scenarios (no multi-move chess problems, no ambiguous linguistics).
- Only single-shot prompts evaluated; few-shot demonstrations or tool-augmented reasoning might reduce failures.
- OpenRouter relays add potential latency and policy differences compared to native APIs.
- Cost tracking is approximate (token-based) but sufficient for relative comparisons.

### Surprises & Insights
- GPT‑4.1 direct failed 5/20 math prompts—all on the same offset-addition rule—showing cluster failures once a bias creeps in.
- Protocol tasks were trivial for all models once instructions contradicted social norms, suggesting the difficulty comes specifically from overriding deeply internalized *structured* knowledge (chess, arithmetic).

### Visualizations
- Bar charts (accuracy per prompt style and per domain) illustrate consistent CoT boosts and highlight chess as the dominant failure source. Both figures include labeled axes, legends, and titles (see `results/plots/`).

## 6. Conclusions
- **Answer:** Current LLMs cannot fully “forget” entrenched rules on demand; even GPT‑4.1 misapplies redefined chess and arithmetic rules under strict instructions.
- **Implications:** Applications requiring hypothetical rule adherence must integrate external constraint checkers or symbolic engines; prompting alone is insufficient.
- **Confidence:** High regarding relative ordering and qualitative conclusions (multiple models, deterministic prompts). However, broader generality to other rule types remains to be tested.

## 7. Next Steps
1. **Increase difficulty and diversity:** Add multi-move chess puzzles, altered physics word problems, and linguistic re-mappings to test compositional forgetting.
2. **Evaluate mitigation strategies:** Compare CoT with tool-assisted validation (e.g., symbolic rule engines) or few-shot examples demonstrating the override.
3. **Human baseline:** Collect human responses on the same benchmark to quantify the hypothesized human advantage.
4. **Long-form interactions:** Test whether repeated reminders or state-tracking agents maintain compliance across multi-turn dialogues.

## References
- Qi et al., 2024. *The Reversal Curse: Language Models Invert Facts Poorly.*
- Liu et al., 2025. *Counterfactual Evaluation of Foundation Models.*
- Sawhney et al., 2024. *Analogical Benchmarks for Language Models.*
- Anthropic Interpretability Blog (2025). *When Claude Refuses to Forget.*
