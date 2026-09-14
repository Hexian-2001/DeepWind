# Reproducibility record

## Published specification

DeepWind-Large is described as an 18-layer, 1024-dimensional decoder-only
Transformer with 16 heads, SwiGLU width 2816, 8 experts, Top-2 routing, patch
size 16, context length 8192, and 21 quantiles. The paper reports AdamW,
`beta1=0.9`, `beta2=0.95`, weight decay 0.01, peak learning rate `1e-4`, 3,000
warmup steps, cosine decay, gradient clipping at 1.0, BF16, global batch 256,
and 100,000 steps.

## Recovered final run

The final `deepwind_large_v5` run reached step 100,000. Its saved configuration
and Slurm log agree on the main dimensions, 8 experts, Top-2 routing, 21-quantile
head, learning rate, global batch, warmup ratio, and step count. The run used
16 distributed accelerator devices (two Setonix nodes, eight visible devices
per node), per-device batch 4, and gradient accumulation 4.

Two material differences are preserved rather than rewritten:

| Setting | Paper | Recovered checkpoint |
|---|---:|---:|
| Rotary/xPOS attention | enabled/described | `use_rotary_emb=false` |
| MoE auxiliary weight | `0.02` | `0.01` |

The current development YAML files also contained later experimental settings
(Student-t heads, 4 experts for Base/Large, and 0.9/0.1 source weighting). They
must not be presented as the paper checkpoint configuration.

## Provenance requirements for every new run

Every run directory must retain:

- resolved Hydra config and override list;
- Git commit and dirty-worktree status;
- Slurm job ID, node list, container digest, Python/package versions;
- random seed, world size, effective global batch, dataset manifest checksum;
- W&B run ID or an offline export;
- train/eval metrics, trainer state, and checkpoint checksums.

No result is publication-ready until a fresh-process load test and at least one
fixed-sample inference regression pass.
