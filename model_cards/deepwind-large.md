---
library_name: transformers
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - wind-power
  - probabilistic-forecasting
  - mixture-of-experts
  - zero-shot
---

# DeepWind-Large

DeepWind-Large is the 1.3B-parameter checkpoint used in "DeepWind: A
foundation model for zero-shot wind power forecasting" (Energy 360, 141793,
2026; DOI: 10.1016/j.energy.2026.141793).

## Intended use

The model produces probabilistic wind-power forecasts from historical power,
available meteorological covariates, and optional site coordinates. It is a
research model for zero-shot and parameter-efficient few-shot evaluation. It is
not a safety-certified dispatch system. Prediction intervals must be validated
and, when needed, recalibrated on the deployment site.

## Architecture

- decoder-only Transformer, 18 layers;
- hidden size 1024, 16 attention heads, SwiGLU width 2816;
- eight sparse experts with Top-2 routing;
- patch size and stride 16, maximum context 8192;
- 21 direct quantiles from 0.01 to 0.99;
- approximately 1.3B total parameters.

## Exact checkpoint configuration

Always use the bundled `config.json`. The released checkpoint records
`use_rotary_emb=false` and `aux_loss_weight=0.01`. These differ from the paper's
method text, which describes RoPE/xPOS and reports 0.02. The checkpoint must not
be loaded with a silently modified configuration.

## Training

The recovered run used AdamW, peak learning rate `1e-4`, 3% linear warmup,
cosine decay, BF16, gradient clipping at 1.0, global batch 256, and 100,000
optimizer steps on Pawsey Setonix. The reported corpus contains approximately
562.4 billion observations from public and permitted wind-energy sources.

## Limitations

- performance may degrade under unseen turbine controls, curtailment policies,
  sensor faults, or covariate conventions;
- the proprietary Shanxi Wind source cannot be redistributed;
- paper prediction intervals showed dataset-dependent under/over-coverage;
- public results should be reproduced with the exact preprocessing and
  WindBench decontamination manifest.

## Files

The public package should contain only `config.json`, the safetensors index,
two safetensors shards, this model card, license/citation metadata, and a SHA-256
manifest. Pickled optimizer states and internal paths are intentionally omitted.
