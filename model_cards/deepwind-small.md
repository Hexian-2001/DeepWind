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

# DeepWind1.0-33M

DeepWind1.0-33M is the ~33M-parameter **small** checkpoint of **DeepWind**, a
decoder-only Transformer foundation model for zero-shot probabilistic wind-power
forecasting, introduced in

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

It is the smallest of the DeepWind family — light enough to run on a laptop GPU
— yet it still issues a 21-quantile forecast directly from a historical power
context and, when available, meteorological covariates and site coordinates.

## Model description

- **Architecture:** decoder-only Transformer with time-aware patching, decoupled
  time/variate attention (RoPE + xPOS), sparse top-2 mixture-of-experts layers,
  and a direct multi-quantile prediction head.
- **Scale:** ~33M trainable parameters (6 layers, hidden 384, 6 heads, SwiGLU
  width 1024, 4 experts).
- **Output:** 21 quantiles (0.01–0.99) per site and horizon.
- **Loss:** trained with a **power-only channel loss** (`channel_loss_weights =
  [1, 0, 0, 0, 0, 0]`), i.e. the objective concentrates gradient on the power
  channel, which is the channel zero-shot evaluation scores.

## Usage

```python
# install the model code first:
#   pip install git+https://github.com/Hexian-2001/DeepWind.git

from src.models.deepwind import DeepWindModel

model = DeepWindModel.from_pretrained("Hexian-2001/DeepWind1.0-33M")
```

Full forecasting (data preparation, expand-collapse decoding, and
post-processing) is provided in the
[DeepWind repository](https://github.com/Hexian-2001/DeepWind)
(`src/inference/`, `infer.py`):

```bash
python infer.py \
  model=deepwind_small \
  inference.checkpoint_path=Hexian-2001/DeepWind1.0-33M \
  data.npy_path=/path/to/site.npy \
  data.metadata_path=/path/to/metadata.csv
```

## Architecture

| Setting | Value |
|---|---|
| hidden size (`d_model`) | 384 |
| layers / heads | 6 / 6 |
| FFN width (`d_ff`) | 1024 (SwiGLU) |
| experts / top-k | 4 / 2 |
| patch size / stride | 16 / 16 |
| max context | 8192 |
| quantiles | 21 (0.01–0.99) |
| normalization | RMSNorm + arcsinh |
| positional encoding | RoPE + xPOS |
| trainable parameters | ~33 M |

## Training

Trained with AdamW (β₁=0.9, β₂=0.95), peak learning rate `1e-4`, 3,000-step
linear warmup, cosine decay, weight decay `0.01`, gradient clipping at `1.0`,
BF16, and a global batch size of 256 for 100,000 steps.

## Evaluation

Zero-shot results on WindBench (8 datasets), as reported in the paper. Each
horizon is the arithmetic mean across the eight datasets:

| Horizon | nCRPS ↓ | nMAE ↓ |
|---|---|---|
| 1 h | 0.0425 | 0.0579 |
| 2 h | 0.0648 | 0.0804 |
| 4 h | 0.0925 | 0.1107 |
| 6 h | 0.1075 | 0.1319 |
| 8 h | 0.1175 | 0.1478 |
| 12 h | 0.1275 | 0.1741 |
| **mean** | **0.0921** | **0.1171** |

## Limitations

This is a research model. Prediction intervals should be validated and
recalibrated on the deployment site; performance may degrade under unseen
turbine controls, curtailment policies, sensor faults, or covariate conventions.
The proprietary Shanxi Wind source cannot be redistributed.

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
