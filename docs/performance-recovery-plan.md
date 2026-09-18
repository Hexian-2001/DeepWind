# DeepWind 模型效果差 — 彻底解决计划

> 状态：诊断完成，Base@10646 early-eval 进行中（job 49523226）
> 目标：让 DeepWind 零样本（zero-shot）先追平/超过 zero-shot TSFM（chronos-2 0.1027 / moirai-2 0.1043 / timesfm-2.5 0.1056），再超过全样本（full-shot）小模型 DeepAR（0.0937）。
> 结论一句话：**架构大概率没问题，问题在「数据分布」+「训练目标与评测目标错位」。先做两个便宜实验定性，再修数据。**

---

## 1. 问题诊断：为什么打不过

### 1.1 证据链（已核实指标，nCRPS 越低越好）

| 模型 | nCRPS | 训练方式 | 规模 |
|---|---|---|---|
| **DeepAR** | **0.0937** | full-shot（每数据集单独训练） | LSTM 128（极小） |
| LightGBM | 0.0996 | full-shot | 树模型 |
| PatchTST | 0.1019 | full-shot | ~几 M |
| **chronos-2** | **0.1027** | zero-shot TSFM | ~710M |
| **moirai-2** | **0.1043** | zero-shot TSFM | ~311M |
| **timesfm-2.5** | **0.1056** | zero-shot TSFM | ~200M |
| dlinear | 0.1087 | full-shot | 线性 |
| nbeats | 0.1109 | full-shot | 小 |
| **DeepWind-Small** | **0.1121** | zero-shot 自训 | 33M |
| ARIMA | 0.1198 | full-shot | — |

三条关键事实：

1. **DeepAR（LSTM-128，参数量比 DeepWind-Small 还小一个量级）赢了所有 zero-shot TSFM** —— 说明在风功率这个任务上，**full-shot 小模型 >> zero-shot 大模型**。规模不是主要矛盾。
2. **DeepWind 输给 chronos/timesfm/moirai 三个「没看过风电数据」的通用 TSFM** —— 说明 DeepWind 的风电专用预训练**反而有害**（学到了错误的先验）。
3. **DeepWind 输给 dlinear/nbeats 这种极简 full-shot 模型** —— 说明 DeepWind 的架构优势（MoE/variate attention/coord embed）在 zero-shot 下没有兑现。

### 1.2 根因：预训练数据 99.44% 是「假数据」

| | 文件数 | 占比 | 通道数 | variate_ids |
|---|---|---|---|---|
| windtoolkit（合成） | 126,588 | 99.44% | 6 | `0,1,2,3,4,5` |
| scada（真实） | 709 | 0.56% | 1–6 | **严重不一致**（见 §4） |

- **windtoolkit 的功率是「用风速+功率曲线算出来的」**，光滑、无噪声、无弃风、无停机、无尾流。模型学到的映射是「power = 功率曲线(wind_speed)」，在合成数据上 loss 极低，但在真实 SCADA 上彻底失效。
- 采样层面 `dataset_weights {windtoolkit:0.7, scada:0.3}` 意味着 **30% 的样本来自只占 0.56% 的 709 个真实风场** → 每个真实文件被重复采样约 53 倍 → 对这 709 个风场**过拟合/记忆**，而不是学到可泛化的风电规律。
- 结论：**零样本失败不是模型小，是「训练分布 ≠ 测试分布」**。测试集 8 个文件里 4 个是真 SCADA（csg_wind_5 / gefc12_wind_7 / gefc14_wind_10 / penmanshiel_15），而训练里真实 SCADA 只有 709 个文件、且 variate 语义乱。

---

## 2. 源码全角落审查报告（对应第 7 点）

审查范围：`src/data/datasets.py`、`src/models/{deepwind,backbone,configuration}.py`、`src/layers/{embeddings,attention,ffn,patch,norm,heads,rope}.py`、`src/losses/criterion.py`、`src/inference/generator.py`、`src/evaluation/{evaluator,reporter,registry}.py`、`src/utils/metrics.py`、`tools/aggregate_eval.py`、数据准备脚本、test/eval metadata。

### 2.1 训练数据（`src/data/datasets.py`）
- `DeepWindTrainDataset` 是 IterableDataset：按 `dataset_weights` 逐样本加权采样，`_ShuffledFileIter` 保证每轮每个文件恰好看到一次，`_generate_from_file` 按 `sample_ratio=0.005` + `[min=4, max=128]` 窗口数采样。逻辑正确。
- `_pad_and_mask` 把通道 pad 到 `max_vars=6`，`channel_mask` 标记真实/填充，`variate_ids` 未提供时用 `pad_val_id=10` 填充。
- **问题 A（variate 语义，见 §4）**：SCADA 的 `variate_ids` 不一致（`0,1,2` / `0,1` / 空 / `0,1,2,4,5,6`…），variate embedding 会被污染。

### 2.2 测评/测试数据（`DeepWindTestDataset` + metadata）
- `DeepWindTestDataset` 按时间轴 0.7/0.1/0.2 切 train/val/test，stride=pred_len，无泄漏。**正确**。
- test 集 8 文件 = 4 windtoolkit + 4 scada。scada 的 variate_ids：csg_wind_5=`0,1,2,4`（**缺 3、多个 4**）、gefc12/14/penmanshiel=`0,1,2`。与训练 SCADA 的 `0,1,2`(407) / `0,1`(186) / `0`(106) 部分对齐，但 csg 的 `0,1,2,4` 布局在训练里几乎不存在。
- **问题 B**：gefc12/gefc14（小时级，序列长度 ~17–19k < 8192）自动回退 `context_length=512`（`datasets.py:774`）。也就是说 **8 个测试集里有 2 个根本没用到 8192 长上下文** —— 8192 对小时级数据本就过量。

### 2.3 模型架构（`backbone.py` / `embeddings.py` / `ffn.py`）
- 交替 time/variate attention（variate 每 2 层一次）、RoPE 只加在 time 层、xPOS、MoE（top-2，含 load-balance 辅助 loss + DDP 哑梯度 sink）、coord embed（经纬度→xyz→d_model）、variate embed（11 个 ID，10=padding）。**实现无明显 bug**（之前已修的 aux-loss/梯度 checkpoint 问题已提交）。
- InstanceNorm（`norm.py`）：每 (batch, variate) 沿时间轴标准化再 arcsinh。`inverse` 用 context 的 loc/scale 还原。**正确**，但注意：arcsinh 压缩了离群大值，可能低估高功率时刻的尾部。

### 2.4 损失/目标（`criterion.py` + `deepwind.py`）— **这是最重要的发现**

```python
targets        = self.embeddings.patcher(embeddings_output.scaled_context)[:, :, 1:]
preds_for_loss = patch_preds[:, :, :-1]
# criterion: pinball loss 在 (B, V, L, P, Q) 上求和，channel_mask 只排除 padding 通道
```

- **问题 C（核心）**：pinball loss 是**对所有 V=6 个通道等权求和**的。`channel_mask` 只把「填充通道」置 0，**并不把功率通道（channel 0）加权**。于是：windtoolkit 样本里功率通道只占 1/6 的梯度；模型把大量容量花在预测风速/风向/温度等「评测根本不看的通道」上。**评测只打 channel 0（power）分**（`evaluator.py:106-114`）。
- **问题 D（目标错位）**：训练是「预测所有通道的下一 patch」，评测是「只预测功率的 1–12h」。这是「多变量自监督预训练」和「单变量功率预测」之间的错位。

### 2.5 推理（`generator.py`）
- greedy 或 MQD（expand-collapse 多轨迹）自回归逐 patch 解码；点预测=中位数分位。**正确**。但自回归逐 patch 前向，12h/5min = 45 步，误差随步长累积（这是 AR 模型通病，非 bug）。

### 2.6 指标（`src/utils/metrics.py` + `reporter.py`）
- nCRPS = 容量归一化的平均 pinball × 2/Q，**与训练损失（pinball）完全对齐**。nMAE/Accuracy/Qualified_Rate 按装机容量归一化，R2 全局。**无 bug**。
- `reporter.py` 先把预测 clip 到 ≥0。Qualified_Rate 阈值按 horizon（≤4h 用 0.15，否则 0.25）。**正确**。
- 唯一注意：nCRPS 按**装机容量**归一化，不是按实际功率，所以低风站点天然更容易拿小 nCRPS —— 跨数据集宏观平均会隐式受容量影响（不是 bug，是约定）。

### 2.7 一个潜在兼容性 bug（记录，非当前主路径）
- `zi_beta` 分布头要求 target ∈ [0,1]（`criterion.py:236-241`），但当前 InstanceNorm 是 **arcsinh(z-score)**，target 会是负值或 >1，**会直接 raise ValueError**。即 `zi_beta` 头在当前 `use_arcsinh=true` 下不可用，除非改成「功率/容量归一化到 [0,1]」+ 关掉 arcsinh。做分布头消融时要注意这点。

---

## 3. 消融实验设计（对应第 1–4 点）

**总原则：先定性（便宜实验定位问题），再定量（修数据/目标重训）。不要一上来就堆规模。**

### 3.0 实验 0（最高优先级，便宜，定性）— Full-shot 天花板测试

用**现成的 `finetune.py` + `DeepWindFinetuneDataset`**，把已训练好的 DeepWind-Small 在 8 个测试集的 train split 上各 fine-tune（full-shot），跑出 nCRPS。

- **若 full-shot DeepWind > DeepAR(0.0937)** → 架构没问题，问题 100% 在 zero-shot 的数据/目标 → 全力修 §4、§2.4。
- **若 full-shot DeepWind ≈ 或 < DeepAR** → 架构本身在风功率任务上就不行 → 走 §7 重设计。

这一步是**整个计划的分水岭**，一天内能出结果，成本极低（复用现有 finetune 链路）。

### 3.1 模型规模（33M vs 3M）—— 结论：**不需要 3M**

- DeepAR（LSTM-128）已经证明**小模型+full-shot 能赢**，规模不是瓶颈。
- 3M 只会让 zero-shot 更欠拟合。**保留 33M Small 作为 zero-shot 工作模型**。
- 只有当 §4 数据修好、目标修好后，若还想验证「规模在 zero-shot 下是否还提升」，才值得用 Base（888M）对比 —— 而这个信号 **Base@10646 early-eval 正在给**（见 §8）。
- **不要**建 3M 配置，除非你只是想快速扫超参（LR/损失权重）——那可以用一个 `deepwind_nano`（d_model 256 / 4 层）只做 2 小时迭代，做完就扔。

### 3.2 MoE vs Dense —— 结论：**控制变量，低优先级**

- MoE 是「容量杠杆」，不是「正确性杠杆」。它不会修数据错位。
- 消融：`use_moe=false`（dense SwiGLU FFN，同 d_ff）重训 Small。**dense ≈ MoE → MoE 是多余复杂度，可以简化；dense < MoE → MoE 有帮助**。
- 时机：**在 §2.4 目标修正之后**再做，否则结论被数据/目标噪声淹没。

### 3.3 预测头 —— 结论：**quantile 头与 nCRPS 对齐，但值得做两个对照**

- 当前 quantile(21)+pinball 与评测 nCRPS **完全对齐**，不是根因。
- 消融项：
  1. **21 vs 9 vs 5 分位**：更少分位更便宜、可能更锐利（但会牺牲 MAE_Coverage 分辨率）。
  2. **quantile vs point-MSE**：point-MSE 头是「纯确定性」对照，用来判断「概率目标是否损害了点精度」（nMAE/Accuracy/R2 只看点预测）。
  3. **quantile vs student_t**：分布头，NLL 目标；但注意 §2.7 的 zi_beta 兼容性坑。
- 时机：低优先级，排在目标修正之后。

### 3.4 context_length —— 结论：**有实际价值，主要为了算力和匹配数据分辨率**

- 当前 8192（5min 数据=28 天，15min=85 天，1h=341 天）。gefc12/14 已回退 512。
- 风功率有强**日周期 + 3–7 天天气尺度周期**。5min 数据上 1–2 天（288–576 步）应该够捕获主要信号；8192 可能过量且**多花 8 倍算力**（5120 token/sample vs 1024 的 640）。
- 消融：**8192 vs 4096 vs 2048 vs 1024**。判据：nCRPS 是否随 context 单调改善？若 2048 ≈ 8192，则永久降到 2048，训练加速 4 倍。
- 注意：context 变短会让 InstanceNorm 的 loc/scale 更噪（norm 在 context 上算），需一起观察。

---

## 4. 预训练数据策略（对应第 5 点）—— **最高杠杆**

### 4.1 调研结论（已核实实机数据）

- 采样概率 `0.7:0.3` 是**样本级**，但**文件级**是 99.44% 合成 / 0.56% 真实 → 真实文件被 53 倍过采样 → 记忆而非泛化。
- SCADA 仅 709 文件、且 variate_ids 五花八门：

| variate_ids | 文件数 | 含义推断 |
|---|---|---|
| `0,1,2,3,4,5` | 126,588 (windtoolkit) | 6 通道：功率+5 个气象量 |
| `0,1,2` | 407 (scada) | 功率+风速+风向 |
| `0,1` | 186 (scada) | 功率+风速 |
| `0` | 106 (scada) | 只有功率 |
| `0,1,2,4,5,6` / `0,1,2,4,6` / `0,1,2,4,5` | 5/4/1 | **variate 6 在 windtoolkit 里根本不存在，且缺 3** |

- **问题 A 详解**：variate embedding 是「一个 variate_id → 一个可学习向量」。如果 windtoolkit 的 variate 3 = 温度、而某 SCADA 源的 variate 4 = 温度，那么 embedding 3 和 4 各自学到「两种物理量的混合」，跨源迁移被污染。csg_wind_5 测试文件用 `0,1,2,4`（缺 3、有 4）正是这个坑。

### 4.2 三个数据问题 & 修复

1. **variate 语义不一致（问题 A）**：建立**统一 variate 本体**（0=功率, 1=轮毂高度风速, 2=风向, 3=温度, 4=气压, 5=空气密度, 6=…），给每个源写一个 `channel → variate_id` 映射，重新生成 metadata。106 个空 variate_ids 的文件补成 `0`。这一步**必须做**，否则 variate embedding 永远是脏的。
2. **sim2real 鸿沟（问题 B）**：windtoolkit 功率是光滑功率曲线，无真实噪声。给合成功率**加真实感增强**：弃风限电（随机上界裁剪）、限功率斜坡、传感器高斯噪声、周期性停机（维护）、尾流损耗扰动。目标：让合成数据的功率分布接近真实 SCADA 的功率分布。
3. **记忆化（问题 C）**：709 个真实文件被 53 倍重采样。方案（按优先级）：
   - **更多真实数据**：核查那 19 个源是否都进来了、是否有被误标/漏标的真实数据。
   - **重平衡**：把 scada 采样权重从 0.3 降到 ~0.1–0.15，同时靠增强扩大 scada 的多样本覆盖（时间平移/加噪，让重复样本不再完全相同）。
   - **cap 每文件窗口数**：真实文件 `max_windows_per_file` 拉高、但配合增强避免记忆。

> 一句话：**问题不是 0.7:0.3 的比例，而是「真实数据太少 + 合成数据太假 + variate 语义乱」**。先修这三样，再谈比例。

---

## 5. Full-shot vs Zero-shot 路线（对应第 6 点）

**双轨并行，不用二选一：**

- **轨道 1（论文卖点，长期）**：修 zero-shot。修 §4 数据 + §2.4 目标 → 重训 Small → 目标 nCRPS < chronos(0.1027) → 再 < DeepAR(0.0937)。
- **轨道 2（立即可做，既是天花板测试也是可发表结果）**：full-shot fine-tune。DeepWind 在每个测试集 train split 上 fine-tune，目标是**超过 DeepAR**，作为「DeepWind-finetuned」上报，同时**定量出架构的 full-shot 上限**。

**决策逻辑**：轨道 2 的结果决定了轨道 1 还值不值得投入。
- full-shot 强 → 架构 OK → 全力修 zero-shot 数据/目标。
- full-shot 弱 → 架构不行 → 直接 §7。

---

## 6. 实验路线图（按杠杆排序，可直接照做）

| 阶段 | 实验 | 成本 | 目的 | 通过判据 |
|---|---|---|---|---|
| **0** | Full-shot 天花板（§3.0） | 1 天，复用 finetune.py | 定性架构 vs 数据 | full-shot nCRPS 是否 > DeepAR |
| **1** | 修 variate 本体 + metadata（§4.2.1） | 半天 | 消除 variate embedding 污染 | metadata 全源统一 |
| **2** | 合成数据真实感增强 + 重平衡（§4.2.2/3） | 1–2 天 | 关 sim2real 鸿沟 + 去记忆 | 合成功率分布 ≈ 真实分布 |
| **3** | 目标修正：loss 只打 channel 0（§2.4） | 改 1 行 + 重训 Small | 聚焦功率预测 | 重训后 zero-shot nCRPS ↓ |
| **4** | context_length 消融（§3.4） | 4 次重训（1024/2048/4096/8192） | 算力 + 匹配分辨率 | 2048 是否 ≈ 8192 |
| **5** | MoE→dense 对照（§3.2） | 1 次重训 | 简化架构 | dense ≈ MoE? |
| **6** | 预测头对照（§3.3） | 3 次重训 | 概率 vs 点 | quantile 是否仍最优 |
| **7** | 规模确认（Base 重训到 full） | 贵 | 最终 zero-shot 上限 | Base > Small 的幅度 |

> 阶段 0–3 是**真正解决「效果差」的核心**，阶段 4–7 是**锦上添花的消融**。先做 0–3。

---

## 7. 兜底：架构重设计（对应第 9 点，仅在阶段 0 失败时触发）

若 full-shot DeepWind 都打不过 DeepAR，说明**decoder-only + next-patch AR + instance-norm 这条路线在风功率任务上本身不合适**，则重设计方向（按优先级）：

1. **非自回归解码**：直接一步输出整个 horizon（像 DeepAR/LightGBM），避免 AR 误差累积。
2. **显式物理先验**：把功率曲线 / 风速→功率单调性作为约束或特征，而不是让模型从合成数据里「背」功率曲线。
3. **换主目标**：pinball→CRPS 直接优化，或加 sharpness 正则。
4. **降低自监督、提高监督**：弱化「预测所有通道」，强化「预测功率」。

（这些是后备，**在阶段 0 给出否定答案之前不动**。）

---

## 8. 当前状态与下一步

- **已出信号（2026-09-18，30651/H1，windtoolkit）**：Base@10646（10.6% 训练）**已全面超过 Small@100000（全量训练）** —— nCRPS 0.0385 vs 0.0435、nMAE 0.0560 vs 0.0628、MAE_Coverage 0.0160 vs 0.0291、R2 0.906 vs 0.885。**结论：规模确实有帮助，且校准（MAE_Coverage）随规模大幅改善。** 但这是合成数据（windtoolkit）的单点；关键考验仍是 SCADA。
- **进行中**：Base@10646 early-eval job 49523226（gpu-dev，nid002184）。注意：Base 的 MQD 推理 ~5.74s/it，比 Small 慢 ~15×，2h 墙钟只够跑 30651 的 H1–H4 左右，**跑不完 48 个 cell**。已拿到方向性信号，**不必**为完整 48-cell Base eval 再排期——因为 §4/§2.4 马上要改数据+目标并重训，届时再给新模型做完整评估更划算。
- **下一步（立即）**：
  1. 等 eval 完成 → 聚合 → 与 Small/TSFM 对比 → 记录。
  2. 启动**阶段 0（full-shot 天花板测试）**——这是最便宜、最能定性的实验。
  3. 并行准备**阶段 1/2（variate 本体 + 数据增强）**。

---

## 附：关键文件索引

- 损失/目标错位：`src/models/deepwind.py:113-131`、`src/losses/criterion.py:94-102`
- 只评测 channel 0：`src/evaluation/evaluator.py:106-114`
- variate 本体：`src/layers/embeddings.py`（`variate_emb`，11 ID，10=padding）
- context 回退：`src/data/datasets.py:774`
- zi_beta 兼容性坑：`src/losses/criterion.py:236-241`
- nCRPS 定义：`src/utils/metrics.py:175-224`
- 数据采样/权重：`src/data/datasets.py:341-373`
