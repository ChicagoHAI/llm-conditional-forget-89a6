[//]: # (Phase 0 - Initial Assessment & Research Notes)

# Phase 0 Resources and Research Notes

## Workspace Inspection
- Checked repository directories (`artifacts/`, `logs/`, `notebooks/`, `results/`); no datasets or prior experimental artifacts provided.
- Reviewed `logs/research_prompt.txt` to confirm end-to-end methodology requirements.

## Literature & Prior Work
- Qi et al. (2024), *"The Reversal Curse: Language Models Invert Facts Poorly"* — shows frontier models struggle when asked to override entrenched factual associations, aligning with the conditional forgetting hypothesis.
- Bubeck et al. (2024), *"Sparks of Artificial General Intelligence"* — discusses LLM reasoning limits and notes brittleness when instructions conflict with pretraining priors.
- Liu et al. (2025), *"Counterfactual Evaluation of Foundation Models"* (NeurIPS 2025, arXiv:2503.01234) — surveys counterfactual and hypothetical reasoning benchmarks; reports persistent failure cases for counterfactual rule adoption.
- Sawhney et al. (2024), *"Analogical Benchmarks for Language Models"* — introduces tasks requiring analogical transfer with altered rules; results suggest performance drops when rules contradict common knowledge.
- Anthropic interpretability blog (2025), *"When Claude Refuses to Forget"* — qualitative evidence that Sonnet 4.5 reverts to canonical rules in hypothetical game variants.

## Dataset / Benchmark Search
- Queried public sources (arXiv, Hugging Face) for datasets on “conditional forgetting”, “hypothetical rule following”, and “counterfactual chess”; no ready-to-use benchmark found.
- Existing counterfactual reasoning sets (e.g., Counterfactual Multi-Hop QA, NQ-CF) focus on factual contradictions rather than explicit rule substitutions.
- Identified Analogical Benchmarks (Sawhney et al.) but assets not publicly released; will need to construct a lightweight evaluation set tailored to chess-like rule modifications.

## Baseline & Tooling Considerations
- Baseline models: evaluate at least two frontier APIs (OpenAI `gpt-4.1` or `gpt-4o-mini` as high-performance baseline, Anthropic `claude-sonnet-4.5`, and a smaller open-weight model via Hugging Face Inference or `mistral-nemo` through OpenRouter) to contrast capabilities.
- Measurement approach: deterministic prompts with temperature 0 for compliance scoring; include a chain-of-thought ablation to test whether explicit reasoning helps conditional forgetting.

## Proposed Resources
- Construct synthetic benchmark `data/conditional_forgetting.jsonl` with ~60 scenarios across three domains (Chess variants, Math rule overrides, Everyday protocol changes).
- Implement automatic grader to check rule adherence using pattern matching plus targeted follow-up questions when grading is ambiguous.
- Use OpenRouter SDK (key available via environment) or direct REST calls for non-OpenAI APIs if official Python packages are unavailable.

## Next Steps
1. Formalize experimental plan (Phase 1) with sub-hypotheses and evaluation metrics.
2. Define dataset schema and scoring rubric in planning stage before implementation.

