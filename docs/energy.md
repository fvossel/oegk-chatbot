# Energy and CO2 per question

Estimate for Ask OEKG / Chat-OEKG as deployed on 2026-11-26: chat model
`Qwen/Qwen3.5-122B-A10B-FP8` on Osnabrück University's LiteLLM gateway, running on
NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB).

## Result

| | per question | 25 questions |
| --- | ---: | ---: |
| **Energy, estimate (EcoLogits method)** | **0.27 Wh** | **6.8 Wh** |
| **CO2e, estimate** | **0.11 g** | **2.8 g** |
| Energy, upper bound (GPUs exclusively busy at full power) | 11.2 Wh | 279 Wh |
| CO2e, upper bound | 4.5 g | 113 g |

For scale: bringing 1 litre of water from 20 °C to the boil takes about 93 Wh
(4.19 kJ/(kg·K) × 80 K). The estimate for 25 questions (6.8 Wh) is enough for
about 70 ml; one question (0.27 Wh) is a 10 W LED bulb lit for about 1.6 minutes.

The previous figure in the paper, 0.115 kWh for 25 requests, was made for the
chat on OpenAI GPT-5. If it was computed with EcoLogits as well, the switch to the
university-hosted model cut the estimate per question by a factor of about 17.

## Measurement

`scripts/energy/measure.py` ran the 25 questions in
[`scripts/energy/questions.txt`](../scripts/energy/questions.txt) through the live
pipeline (fresh answers, no cache) on 2026-11-26 and recorded every chat-model call:
prompt tokens, generated tokens (reasoning included, as reported by the gateway)
and the call's wall time. A question that triggered a clarification was answered
again with the first option, as a user would. The raw data are in
[`scripts/energy/runs-2026-11-26.jsonl`](../scripts/energy/runs-2026-11-26.jsonl).

| | mean | median |
| --- | ---: | ---: |
| Chat-model calls per question | 2.2 | 2 |
| Generated tokens per question | 1707 | 849 |
| Prompt tokens per question | 20 300 | 14 400 |
| Chat-model time per question | 17.9 s | 10.9 s |

Most generated tokens are the reasoning of SPARQL generation; questions that need
a repair or a retry on an empty result cost two to four times as much (up to
5500 tokens). Summaries, explanations and the ambiguity check run without
reasoning and cost a few hundred tokens each. Embeddings (`bge-m3`, 0.57 B
parameters, a few dozen tokens per question) are left out as negligible.

## Method

`scripts/energy/estimate.py` computes both numbers from the measurement; every
assumption below is a command-line flag.

**Estimate.** The EcoLogits LLM inference methodology
(<https://ecologits.ai/latest/methodology/llm_inference/>):

- GPU energy per generated token and GPU:
  f_E = α · e^(β·B) · P_active + γ with α = 1.17·10⁻⁶, β = −1.12·10⁻², γ = 4.05·10⁻⁵
  (Wh, P_active in billions), batch size B = 64. For P_active = 10 B this is
  0.046 mWh.
- GPUs per model instance: memory 1.2 × 122 B parameters × 1 byte (FP8) =
  146 GB, on 96 GB GPUs → 2 GPUs (rounded up to a power of two).
- Non-GPU server share: 1.2 kW per 8-GPU server, attributed by GPU count and
  divided by the batch size, over the measured call time.
- Data centre overhead: PUE 1.5, assumed. The university's value is not known to
  us; 1.5 is deliberately conservative next to the 1.09–1.20 that large cloud
  providers report (EcoLogits' provider table).
- Grid: 406 g CO2e/kWh, the German electricity mix of 2025 including upstream
  chains, preliminary value (Umweltbundesamt, *Entwicklung der spezifischen
  Treibhausgas-Emissionen des deutschen Strommix in den Jahren 1990–2025*,
  Climate Change 16/2026, Table 1). The direct CO2 factor alone is 344 g/kWh.

**Upper bound.** Each call is taken to keep its 2 GPUs at their full 600 W
(NVIDIA's maximum for the RTX PRO 6000 Blackwell Server Edition) plus the 300 W
non-GPU share for its whole measured duration, with nobody else's requests in the
same batch, times the PUE. Real use is far below this: the gateway batches
requests of many users onto the same GPUs.

## Limits

- No request can be metered on shared GPUs; both numbers are models, not readings.
  Measured power and throughput from the gateway's operators would replace both
  (`--gpu-watts`, `--pue`, or a per-token value).
- EcoLogits counts generated tokens only. Our prompts are large (about 12 000
  tokens per SPARQL generation), so the prompt processing the estimate leaves out
  is not negligible here; the upper bound includes it.
- The EcoLogits fit was made on H100 GPUs. The RTX PRO 6000 has less memory
  bandwidth, so it generates tokens more slowly per GPU, which tends to raise the
  energy per token.
- Values hold for these 25 questions; long answers that need several repairs
  cost up to about three times the mean (0.88 Wh for the costliest question).
- If the university buys certified green electricity, the market-based CO2e is
  lower; the figures above use the location-based German mix.

## Text proposals

Paper (replaces the GPT-5 figure):

> We estimated the energy use of Chat-OEKG with the EcoLogits methodology [ref],
> based on the token counts measured for 25 representative questions. Running on
> university-hosted GPUs, a question uses about 0.27 Wh on average (6.8 Wh for all
> 25), which corresponds to about 0.11 g CO2e at the 2025 German electricity mix
> (406 g CO2e/kWh [UBA ref]). Even if every request occupied its two GPUs alone at
> full power, a question would stay below 11.2 Wh.

App disclaimer (one line):

> Answering a question uses about 0.3 Wh of electricity on the university's
> servers, roughly what a 10 W LED bulb uses in two minutes.
