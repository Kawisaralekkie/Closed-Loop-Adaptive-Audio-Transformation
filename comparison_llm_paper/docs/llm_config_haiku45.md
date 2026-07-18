# LLM Configuration & Run Profile — Claude Haiku 4.5

Reference sheet for the thesis "Methodology / Experimental Setup" section.
Empirical per-configuration numbers (tokens, cost, latency) are generated into
`plots/comparison_llm/llm_run_profile.csv` by `scripts/plot_comparison_llm_v2.py`.

## 1. Model & decoding parameters
Source: `src/agents/llm_privacy_control_agent.py::_call_bedrock`

| Setting | Value |
|---|---|
| Provider | Amazon Bedrock (`bedrock-runtime`, region `ap-southeast-1`) |
| Model ID | `global.anthropic.claude-haiku-4-5-20251001-v1:0` |
| API | Bedrock **Converse** API with tool use |
| Temperature | **0.1** (near-deterministic; favours consistent recipe choices) |
| max_tokens | **512** per call |
| Guardrail | Bedrock Guardrail `urban-soundscape-guardrail` (id `13t27m2qd7yx`, version `DRAFT`), PII + topic filters, enabled by default |

## 2. Prompt
- System prompt: `SYSTEM_PROMPT` in `src/agents/llm_privacy_control_agent.py`
  (role, available tools, decision + source-separation guidelines, privacy thresholds).
- Tools exposed to the model (`toolSpec`): `apply_midband_attenuation`,
  `apply_strong_blur`, `apply_source_separation_blur`.
- Per-chunk user message includes: `speech_ratio`, `vad_confidence_mean`,
  `num_speech_segments`, and (for the memory variant) the ExperienceMemory summary.

## 3. Pricing (for cost calculation)
Amazon Bedrock — Claude Haiku 4.5 (USD per 1M tokens):

| | Input | Output |
|---|---|---|
| **Bedrock** (used here) | **$1.10** | **$5.50** |
| Anthropic direct (ref.) | $1.00 | $5.00 |

Sources: [Amazon Bedrock Claude Haiku 4.5 listing](https://futureagi.com/llm-cost-calculator/bedrock/us-anthropic-claude-haiku-4-5-20251001-v1-0),
[Anthropic API pricing](https://www.cloudzero.com/blog/claude-api-pricing/).
*(Content rephrased for compliance with licensing restrictions.)*

Cost per configuration = `input_tokens/1e6 × 1.10 + output_tokens/1e6 × 5.50`
(constants `HAIKU45_PRICE_IN_PER_MTOK` / `..._OUT_PER_MTOK` in the plot script).

## 4. Where the empirical numbers come from (per chunk / per run)
- Tokens: report `total_llm_token_usage` and per-chunk `llm_token_usage`
  (`input_tokens`, `output_tokens`, `total_tokens`).
- Latency: per-call `llm_responses[].latency_ms` → mean / p50 / p95.
- Trials: per-chunk `trials` (mean trials = a proxy for decision efficiency;
  the memory variant is expected to need fewer trials over time).

Run `python3 scripts/plot_comparison_llm_v2.py` → produces
`plots/comparison_llm/llm_run_profile.csv` with columns:
`config, model, temperature, max_tokens, n_llm_calls, speech_chunks,
mean_trials, input_tokens, output_tokens, total_tokens, cost_usd,
cost_per_speech_chunk_usd, mean_latency_ms, p50_latency_ms, p95_latency_ms`.
