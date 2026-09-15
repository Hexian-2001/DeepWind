# Reproducibility record

## Published specification

DeepWind-Large is described as an 18-layer, 1024-dimensional decoder-only
Transformer with 16 heads, SwiGLU width 2816, 8 experts, Top-2 routing, patch
size 16, context length 8192, and 21 quantiles. The paper reports AdamW,
`beta1=0.9`, `beta2=0.95`, weight decay 0.01, peak learning rate `1e-4`, 3,000
warmup steps, cosine decay, gradient clipping at 1.0, BF16, global batch 256,
and 100,000 steps.

## Recovered final run

> Historical note (pre-retrain): this describes the legacy `deepwind_large_v5`
> checkpoint as recovered before the 2026-09 retrain campaign. The retrain
> (Small/Base/Large to paper spec) uses `use_rotary_emb=true` and
> `aux_loss_weight=0.02`, matching the paper. See the leaderboard (below) for
> the authoritative per-model configuration.

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

---

# Model registry & leaderboard

The sections below describe how a completed train → eval run is registered into
a **git-committed JSONL leaderboard** (`results/leaderboard.jsonl`), and how to
compare, export paper tables, and run seed sweeps.

## 1. 理念

- **源真相 = `results/leaderboard.jsonl`**（append-only、一行一个被评估模型、随仓库 git 提交，
  可 code-review、可 diff、可追溯）。wandb 与 `run_info.json` 是它的上游原材料。
- **一行 = 一个模型**：架构 + 训练超参 + seed + git commit + 全部指标（宏观 / 逐数据集 / 逐时长）。
- **config_hash 归组设计**：同一设计（架构+训练+数据）不同 seed 的哈希相同 →
  一个 seed sweep 自动归为同一 family，便于求 mean±std。

## 2. 流水线

```
训练 (train_paper.sbatch)
   └─> 评估 (eval_paper.sbatch + tools/aggregate_eval.py)
          └─> 注册 (tools/register_eval.py)          # 写入 results/leaderboard.jsonl
                 ├─> 对比 (tools/compare_models.py)
                 └─> 论文表 (tools/export_report.py)
```

### 2.1 训练

```bash
MODEL=small sbatch --export=ALL,MODEL=small scripts/setonix/train_paper.sbatch
```

run name 固定为 `deepwind-small-paper-seed42`（`--requeue` 时同名 `output_dir` 自动断点续训，
跨 24h 墙钟限制）。

### 2.2 评估

```bash
CKPT=/scratch/pawsey0115/hwang4/projects/deepwind/runs/DeepWind-Research/deepwind-small-paper-seed42/checkpoints/checkpoint-100000 \
  sbatch --export=ALL,MODEL=small,CKPT=... scripts/setonix/eval_paper.sbatch
# 完成后
python tools/aggregate_eval.py <eval_dir>
```

### 2.3 注册（把一次评估写进排行榜）

```bash
python tools/register_eval.py \
  /scratch/pawsey0115/hwang4/projects/deepwind/runs/results/deepwind/eval-small-paper \
  --model-id deepwind-small-paper-seed42 --variant small --tags paper-spec,baseline,seed42
```

`register_eval.py` 会自动读取：

| 来源 | 取到的字段 |
|------|-----------|
| `<eval_dir>/run_info.json` | `checkpoint_path`、评估时 git commit、`run_name` |
| `checkpoints/run_info.json`（训练时快照） | `config.model`（架构）、`config.training`（训练）、`config.seed`、`config.data.seed`、训练 git commit |
| `<eval_dir>/_aggregate.json` | 宏观均值 + 逐时长 nCRPS |
| `<eval_dir>/*/H_*.json` | 逐数据集指标 |
| `checkpoint-*/trainer_state.json` | `global_step` + final loss |

## 3. 排行榜 schema

`results/leaderboard.jsonl` 每行一个 JSON：

```json
{
  "model_id": "deepwind-small-paper-seed42",
  "variant": "small",
  "checkpoint": "…/checkpoints/checkpoint-100000",
  "eval_name": "eval-small-paper",
  "eval_dir": "…/results/deepwind/eval-small-paper",
  "git_commit": "3a0ef04…",
  "training_git_commit": "5e167d3…",
  "created_at": "2026-09-15T…Z",
  "config_hash": "sha256…",
  "architecture": { "d_model": 384, "num_layers": 6, "…": "…" },
  "training":     { "learning_rate": 1e-4, "per_device_train_batch_size": 8, "…": "…" },
  "seed": 42,
  "data_seed": null,
  "metrics": {
    "mean": { "nCRPS": 0.1121, "nMAE": 0.1482, "Accuracy": 0.7814,
              "Qualified_Rate": 0.7327, "MAE_Coverage": 0.1085, "R2": 0.5268,
              "mean_wQuantileLoss": 0.3120 },
    "per_dataset": { "30651": { "nCRPS": 0.1083, "…": "…" }, "…": "…" },
    "per_horizon_nCRPS": { "nCRPS_H1": 0.0567, "…": "…" },
    "n_cells": 48
  },
  "training_outcome": { "global_step": 100000, "final_loss": 0.0458 },
  "tags": ["paper-spec", "baseline", "seed42"]
}
```

指标好坏方向（`compare_models.py` / `export_report.py` 用来自动标胜者）：

- **越低越好**：`nCRPS`、`nMAE`、`MAE_Coverage`、`mean_wQuantileLoss`
- **越高越好**：`Accuracy`、`Qualified_Rate`、`R2`

## 4. 对比

```bash
python tools/compare_models.py --all
python tools/compare_models.py --variant small,base,large
python tools/compare_models.py deepwind-small-paper-seed42 deepwind-base-paper-seed42
python tools/compare_models.py --family <config_hash> --group-family   # 同设计多 seed → mean±std
```

输出：宏观指标表 + 逐数据集 nCRPS/nMAE 表 + 逐时长 nCRPS，每列 ` <--` 标注该行最优。

## 5. 导出论文表

```bash
python tools/export_report.py --format latex    --out results_table.tex
python tools/export_report.py --format markdown --variant small,base,large
python tools/export_report.py --format csv      --all --out results_table.csv
```

LaTeX 输出为 booktabs 表格（`\toprule`/`\midrule`/`\bottomrule`），最优值自动 `\textbf{}` 加粗。

## 6. Seed sweep

```bash
python tools/run_seed_sweep.py --model small --seeds 42,43,44 --dry-run
python tools/run_seed_sweep.py --model base  --seeds 42,43,44
```

每个 seed 提交一个独立训练（`DEEPWIND_RUN_NAME=deepwind-<model>-paper-seed<s>`，
`DEEPWIND_TRAIN_OVERRIDES=seed=<s>` 走 Hydra 顶层 `seed` override，`train.py` 里
`set_seed(cfg.seed)` 会用它重设模型初始化 / 数据顺序 / RNG）。

每个 seed 训完→评估→`register_eval.py` 注册；因 `config_hash` 不含 seed，多 seed 自动归组，
`compare_models.py --group-family` 可直接出 mean±std。

> 已知限制：`train_paper.sbatch` 里 `wandb.tags=[...,seed42]` 是写死的，seed sweep 的 wandb tag
> 仍显示 `seed42`；真实 seed 以排行榜 `seed` 字段与 `wandb.notes`/`run_name` 为准。

## 7. config_hash 语义

```
config_hash = sha256( canonical_json({ model, training', data', data_eval }) )
```

其中从 `training` 剔除 **run 身份字段**（`run_name`/`output_dir`/`report_to`/`overwrite_output_dir`）
与 **汇报/checkpoint 旋钮**（`logging_*`、`save_*`、`eval_*`、`load_best_model_at_end`、
`ddp_find_unused_parameters`、`prediction_loss_only` 等），并从 `data` 剔除 `seed`。
**seed 不参与哈希** → 同设计不同 seed 得到相同 `config_hash`，归为同一 family。

## 8. 复用与边界

- 本体系**只读** `run_info.json`，不重复采集 provenance（`src/utils/provenance.py` 已做）。
- `reproduction/` 里既有的论文图表脚本（reliability / coverage / efficiency）继续读
  `<eval>/<dataset>/H_*.json`，与本体系**互补不冲突**——本体系是它们的上游（注册表/对比/论文表）。
- 排行榜存**绝对 /scratch 路径**（诚实、可直接复现），通过 `DEEPWIND_RUNS_ROOT` 重定向。
