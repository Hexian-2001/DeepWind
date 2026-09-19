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

# DeepWind1.0-1.3B

DeepWind1.0-1.3B is the ~1.3B-parameter **large** checkpoint of **DeepWind**, a
decoder-only Transformer foundation model for zero-shot probabilistic wind-power
forecasting, introduced in

> H. Wang et al., "DeepWind: A foundation model for zero-shot wind power
> forecasting," *Energy*, vol. 360, 141793, 2026.
> https://doi.org/10.1016/j.energy.2026.141793

This is the recovered final paper checkpoint — the largest member of the family.
It issues a 21-quantile forecast directly from a historical power context and,
when available, meteorological covariates and site coordinates.

## Model description

- **Architecture:** decoder-only Transformer with time-aware patching, decoupled
  time/variate attention (xPOS), sparse top-2 mixture-of-experts layers, and a
  direct multi-quantile prediction head.
- **Scale:** ~1.3B trainable parameters (18 layers, hidden 1024, 16 heads,
  SwiGLU width 2816, 8 experts).
- **Output:** 21 quantiles (0.01–0.99) per site and horizon.

## Usage

```python
# install the model code first:
#   pip install git+https://github.com/Hexian-2001/DeepWind.git

from src.models.deepwind import DeepWindModel

model = DeepWindModel.from_pretrained("Hexian-2001/DeepWind1.0-1.3B")
```

Full forecasting (data preparation, expand-collapse decoding, and
post-processing) is provided in the
[DeepWind repository](https://github.com/Hexian-2001/DeepWind)
(`src/inference/`, `infer.py`):

```bash
python infer.py \
  model=deepwind_large \
  inference.checkpoint_path=Hexian-2001/DeepWind1.0-1.3B \
  data.npy_path=/path/to/site.npy \
  data.metadata_path=/path/to/metadata.csv
```

## Architecture

| Setting | Value |
|---|---|
| hidden size (`d_model`) | 1024 |
| layers / heads | 18 / 16 |
| FFN width (`d_ff`) | 2816 (SwiGLU) |
| experts / top-k | 8 / 2 |
| patch size / stride | 16 / 16 |
| max context | 8192 |
| quantiles | 21 (0.01–0.99) |
| normalization | RMSNorm + arcsinh |
| positional encoding | xPOS (RoPE disabled in the recovered config) |
| trainable parameters | ~1.3 B |

> The recovered checkpoint records `use_rotary_emb=false` and
> `aux_loss_weight=0.01`, which differs from the paper text's RoPE/xPOS and
> 0.02; see the repository's reproducibility notes.

## Training

Trained with AdamW (β₁=0.9, β₂=0.95), peak learning rate `1e-4`, 3,000-step
linear warmup, cosine decay, weight decay `0.01`, gradient clipping at `1.0`,
BF16, and a global batch size of 256 for 100,000 steps.

## Evaluation

Zero-shot results on WindBench (8 datasets), as reported in the paper. Each
horizon is the arithmetic mean across the eight datasets:

| Horizon | nCRPS ↓ | nMAE ↓ |
|---|---|---|
| 1 h | 0.0400 | 0.0563 |
| 2 h | 0.0638 | 0.0784 |
| 4 h | 0.0920 | 0.1055 |
| 6 h | 0.1013 | 0.1223 |
| 8 h | 0.1004 | 0.1360 |
| 12 h | 0.1030 | 0.1572 |
| **mean** | **0.0834** | **0.1093** |

Note the non-monotonic scaling: Large edges ahead of Base at the long horizons
(8–12 h) but does not dominate across the board — Base remains the recommended
default.

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
