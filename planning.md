## Research Question
Can contemporary large language models correctly override entrenched knowledge and apply newly specified rules (“conditional forgetting”) across multiple domains such as chess variants, arithmetic rewrites, and everyday procedures?

## Background and Motivation
- Anecdotal reports and recent studies (Qi et al., 2024) highlight that LLMs frequently revert to memorized facts even when explicitly told otherwise.
- Conditional forgetting is crucial for hypothetical reasoning, counterfactual planning, and interactive learning; failure here limits agent safety and alignment.
- Prior work largely measures factual reversals or counterfactual QA rather than explicit rule substitution, leaving a gap this study aims to fill.

## Hypothesis Decomposition
1. **H1 (Rule Adoption Accuracy):** When tasked with applying newly specified rules that contradict prior knowledge, LLMs achieve <80% compliance accuracy.
2. **H2 (Chain-of-Thought Effect):** Providing explicit chain-of-thought (CoT) instructions improves compliance relative to terse answers, but the improvement is modest (<15 percentage points).
3. **H3 (Model Variation):** Larger, frontier models (OpenAI `gpt-4.1`, Anthropic `claude-sonnet-4.5`) outperform a strong open-weight baseline (`mistral-large-2024` via OpenRouter) but still fall short of perfect compliance (statistically significant gap vs. 100%).

## Proposed Methodology

### Approach
Create a targeted benchmark of conditional forgetting scenarios, query multiple LLMs with tightly controlled prompts, auto-score responses for rule adherence, and conduct statistical comparisons across model types and prompting strategies.

### Experimental Steps
1. **Benchmark Construction:** Author ~60 scenarios grouped into three domains (Chess Rule Overrides, Arithmetic Rule Overrides, Protocol/Policy Changes) with deterministic gold answers. *Rationale:* Balanced coverage of symbolic, numeric, and procedural knowledge.
2. **Prompt Template Design:** Develop base prompts (Direct Answer) and CoT prompts (request reasoning before answer) ensuring consistent instruction structure. *Rationale:* Controls for prompt-induced variance.
3. **API Harness Implementation:** Build Python scripts using `openai` and OpenRouter REST calls (for Anthropic + Mistral) with retry, logging, and deterministic parameters (`temperature=0`). *Rationale:* Guarantees reproducibility and auditability.
4. **Response Collection:** Query each model/prompt pair over the full benchmark, caching raw outputs in `results/model_outputs/*.jsonl`. *Rationale:* Enables later re-analysis without re-querying APIs.
5. **Automated Scoring:** Implement grader comparing model outputs to gold answers, with regex-based normalization and rule-specific check functions; log pass/fail plus error categories. *Rationale:* Objective and scalable evaluation.
6. **Statistical Analysis:** Compute accuracy, Wilson 95% confidence intervals, perform McNemar tests between prompt variants per model, and binomial tests vs. perfect compliance. *Rationale:* Quantify uncertainty and significance.
7. **Error Analysis:** Sample failing cases per domain, categorize (rule reversion, partial compliance, hallucinated rules). *Rationale:* Understand failure modes and support qualitative claims.

### Baselines
- `gpt-4.1` Direct (no CoT) — high-end proprietary baseline.
- `claude-sonnet-4.5` Direct — cross-provider check.
- `mistral-large-2024` Direct — strong open-weight baseline via API.
- CoT variants for each model for within-model comparison.

### Evaluation Metrics
- **Compliance Accuracy:** Fraction of scenarios where final answer satisfies substituted rules. Primary metric aligned with hypothesis.
- **Domain-specific Accuracy:** Accuracy per domain to reveal differential difficulty.
- **Error Taxonomy Counts:** Frequency of each error type for qualitative insights.
- **Latency & Cost (secondary):** Track tokens/cost to contextualize feasibility.

### Statistical Analysis Plan
- Compute Wilson score confidence intervals (95%) for accuracy estimates.
- Use binomial tests (H0: accuracy = 1.0) to confirm significant deviation from perfect compliance.
- Apply McNemar’s test for paired comparisons between Direct vs. CoT per model (α = 0.05, adjust via Holm-Bonferroni across three models).
- Report effect sizes (Cohen’s h) for accuracy differences.

## Expected Outcomes
- Support for hypothesis if all models exhibit statistically significant deviation from perfect compliance and notable rule-reversion errors.
- Partial refutation if any model achieves ≥95% accuracy with non-overlapping confidence intervals vs. others.
- Insight into whether CoT meaningfully assists conditional forgetting tasks.

## Timeline and Milestones
- **Benchmark design & tooling:** 1.5 hours
- **API harness & data collection:** 1.0 hour (staggered requests with caching)
- **Scoring & analysis scripts:** 1.0 hour
- **Experiments & analysis:** 1.0 hour
- **Documentation & validation:** 1.0 hour
- Buffer (debugging): embedded flex (~30 minutes) within above allocations; total fits within available session time.

## Potential Challenges
- **API Access Limitations:** Mitigation via caching outputs and fallback models if specific endpoints unavailable.
- **Automatic Grading Ambiguity:** Address by designing canonical answer formats and normalizing responses; manual review for edge cases.
- **Cost Management:** Limit benchmark size to 60 scenarios and monitor token usage per call.
- **Time Constraints:** Prioritize chess and arithmetic domains if dataset creation runs long; document any scope reductions.

## Success Criteria
- Complete dataset, execution scripts, and reproducible pipeline within session.
- Obtain statistically grounded accuracy estimates for ≥3 model/prompt combinations.
- REPORT.md documents methodology, quantitative results, and failure analysis with supporting artifacts.

