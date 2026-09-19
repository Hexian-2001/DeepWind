---
library_name: transformers
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - wind-power
  - probabilistic-forecasting
  - mixture-of-experts
  - zero-shot
  - foundation-model
---

# DeepWind1.0-890M

DeepWind1.0-890M is the 888M-parameter base checkpoint of **DeepWind**, a
decoder-only Transformer foundation model for zero-shot probabilistic
wind-power forecasting, introduced in

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

The model is pre-trained on ~562 billion wind observations and issues a
21-quantile forecast directly from a historical power context and, when
available, meteorological covariates and site coordinates.

## Model description

- **Architecture:** decoder-only Transformer with time-aware patching,
  decoupled time/variate attention (RoPE + xPOS), sparse top-2
  mixture-of-experts layers, and a direct multi-quantile prediction head.
- **Scale:** 888.24M trainable parameters (12 layers, hidden 1024, 16 heads,
  SwiGLU width 2816, 8 experts).
- **Output:** 21 quantiles (0.01–0.99) per site and horizon.

## Usage

```python
# install the model code first:
#   pip install git+https://github.com/Hexian-2001/DeepWind.git

from src.models.deepwind import DeepWindModel

model = DeepWindModel.from_pretrained("Hexian-2001/DeepWind1.0-890M")
```

Full forecasting (data preparation, expand-collapse decoding, and
post-processing) is provided in the
[DeepWind repository](https://github.com/Hexian-2001/DeepWind)
(`src/inference/`, `infer.py`).

## Architecture

| Setting | Value |
|---|---|
| hidden size (`d_model`) | 1024 |
| layers / heads | 12 / 16 |
| FFN width (`d_ff`) | 2816 (SwiGLU) |
| experts / top-k | 8 / 2 |
| patch size / stride | 16 / 16 |
| max context | 8192 |
| quantiles | 21 (0.01–0.99) |
| normalization | RMSNorm + arcsinh |
| positional encoding | RoPE + xPOS |
| trainable parameters | 888.24 M |

## Training

Trained with AdamW (β₁=0.9, β₂=0.95), peak learning rate `1e-4`, 3% linear
warmup, cosine decay, weight decay `0.01`, gradient clipping at `1.0`, BF16,
and a global batch size of 256.

## Limitations

This is a research model. Prediction intervals should be validated and
recalibrated on the deployment site; performance may degrade under unseen
turbine controls, curtailment policies, sensor faults, or covariate
conventions. The proprietary Shanxi Wind source cannot be redistributed.

## Citation

```bibtex
@article{wang2026deepwind,
  title     = {DeepWind: A foundation model for zero-shot wind power forecasting},
  author    = {Wang, Hexian and Zhou, Tongming and Jia, Chengzhen and Liu, Yushan and Wang, Lingmei},
  journal   = {Energy},
  volume    = {360},
  pages     = {141793},
  year      = {2026},
  doi       = {10.1016/j.energy.2026.141793}
}
```

## License

Apache-2.0. See the
[DeepWind repository](https://github.com/Hexian-2001/DeepWind) for the code
license and `CITATION.cff`.
