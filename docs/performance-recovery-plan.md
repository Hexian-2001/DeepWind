# DeepWind 性能修复计划（Phase 0 起：目标 + 数据修正，再消融）

> 状态：诊断完成；Base@10646 已评估并入库（`deepwind-base-10646`，nCRPS 0.0961，全场第 2）。
> 一句话结论：**架构没问题、规模有用；短板是「训练目标与评测目标错位」+「训练数据分布与真实数据错位」。先修这两样（Phase 0/1），再消融其余（Phase 2+）。**
> 本版编号：**Phase 0 = loss 修正**。旧的「Phase 0 full-shot 天花板测试」已被 Base 结果取代，不再作为前置步骤（详见 §1.2 决策说明）。

---

## 0. 证据与诊断

### 0.1 当前排行榜（nCRPS 越低越好，源：`results/leaderboard.jsonl`）

| 排名 | 模型 | nCRPS | 训练方式 |
|---|---|---|---|
| 1 | DeepAR | 0.0923 | full-shot（LSTM-128） |
| **2** | **DeepWind-Base@10646（10.6% 训练）** | **0.0961** | zero-shot |
| 3 | LightGBM | 0.0982 | full-shot |
| 4 | chronos-2 | 0.1013 | zero-shot TSFM |
| 5 | PatchTST | 0.1015 | full-shot |
| 6 | moirai-2 | 0.1027 | zero-shot TSFM |
| 7 | timesfm-2.5 | 0.1042 | zero-shot TSFM |
| 8 | DLinear | 0.1074 | full-shot |
| 9 | N-BEATS | 0.1089 | full-shot |
| 10 | DeepWind-Small@100k | 0.1097 | zero-shot |
| 11 | ARIMA | 0.1174 | full-shot |

### 0.2 Base@10646 宏观指标 —— 唯一的短板

| 指标 | Base@10646 | DeepAR | Small@100k | 说明 |
|---|---|---|---|---|
| nCRPS ↓ | 0.0961 | **0.0923** | 0.1097 | 差 DeepAR 4% |
| nMAE ↓ | **0.1211** 🥇 | 0.1359 | 0.1453 | 全场第一 |
| Accuracy ↑ | **0.8097** 🥇 | 0.8076 | 0.7838 | 全场第一 |
| Qualified_Rate ↑ | **0.8001** 🥇 | 0.7681 | 0.7403 | 全场第一 |
| R2 ↑ | 0.6430 | **0.6504** | 0.5453 | 全场第二 |
| **MAE_Coverage ↓** | **0.1229** ⚠️ | 0.0444 | 0.1082 | **唯一烂的指标：比 DeepAR 差 3×，比 moirai-2(0.0224) 差 5×** |

**解读**：点预测能力（nMAE / Accuracy / Qualified_Rate）已经全场第一；唯一硬伤是**校准**（MAE_Coverage）。这是「训练数据太光滑 → 模型过分自信 → 区间覆盖差」的典型症状，同时「loss 平摊 6 通道」稀释了 power 的梯度。这两点分别对应 Phase 1 和 Phase 0。

### 0.3 三条根因（→ 对应修复 Phase）

1. **目标错位（→ Phase 0）**：pinball loss 在 6 个通道上等权求和，power（channel 0）只拿 1/6 梯度；但评测只给 power 打分。（`src/losses/criterion.py`、`src/models/deepwind.py:113-131`）
2. **数据分布错位（→ Phase 1）**：训练数据 99.44% 是合成 windtoolkit（126,588 份，功率 = 风速的光滑功率曲线，无弃风/停机/噪声）；真实 SCADA 仅 709 份（0.56%），还被样本级权重 `{windtoolkit:0.7, scada:0.3}` 过采样约 53× → 记忆而非泛化。
3. **variate 本体不一致（→ Phase 2）**：SCADA 的 `variate_ids` 五花八门（`0,1,2` / `0,1` / `0` / `0,1,2,4,5,6`…），污染 variate embedding。

---

## 1. 路线图（本版编号）

| Phase | 实验 | 治什么 | 成本 | 通过判据 |
|---|---|---|---|---|
| **0** | loss 加权 / 只打 power | 目标错位 | Small 重训 1 次 | MAE_Coverage ↓ 且 nCRPS 不涨 |
| **1** | 数据 sim2real 增广 + 重平衡 | 分布错位 | 审计 + Small 重训 | nCRPS ↓（往 0.09 走） |
| **2** | 统一 variate 本体 | embedding 污染 | 半天（改 metadata） | 全源变体语义统一 |
| **3** | context_length（1024/2048/4096/8192） | 算力 / 分辨率 | 4 次重训 | 2048 ≈ 8192? |
| **4** | MoE → dense | 简化架构 | 1 次重训 | dense ≈ MoE? |
| **5** | 预测头（quantile / MSE / student_t） | 概率 vs 点 | 3 次重训 | quantile 仍最优? |
| **6** | 规模确认（Base/Large 用胜出配置重训） | 最终上限 | 贵 | Base/Large > Small |

> Phase 0/1 是**解决「效果差」的核心**；Phase 2 是必要的数据卫生；Phase 3–5 是锦上添花的消融；Phase 6 是最终重训。

### 1.1 决策树（Phase 0/1 怎么串）

- **Phase 0 有用**（MAE_Coverage ↓ 且 nCRPS 不涨）→ 把新 loss **锁死**为 baseline；Phase 1 在**新 loss 之上**测（用对的 loss，绝不退回等权 loss）→ 组合验证 → Phase 6 重训 Base/Large。
- **Phase 0 没用** → 说明「power 拿 1/6 梯度」不是 MAE_Coverage 烂的根因；病在数据分布 → 直接主攻 Phase 1；loss 暂维持现状。
- 两条路都收敛到同一终点：**新 loss + 新数据**在 Small 验证通过 → Phase 6 重训 Base/Large。

### 1.2 为什么不再把 full-shot 当 Phase 0

full-shot 天花板测试的唯一作用是回答「架构行不行」。**Base@10646（zero-shot、只训 10.6%）已经 nCRPS 0.0961、全场第 2、只差 DeepAR 4%，且唯一烂指标 MAE_Coverage 能被「loss 平摊 + 数据光滑」完全解释**——这已经回答了「架构没问题」。所以 full-shot 的决策价值已用完，降级为「可选后补」（写论文要天花板数字时再跑），不再阻塞主线。

---

## 2. Phase 0 —— loss 修正（详）

### 2.1 是什么
训练 pinball loss 在 `(B, V, L, P, Q)` 上等权求和（V=6 通道），power 只拿 1/6 梯度；评测只看 channel 0。

### 2.2 怎么改（已实现，config 开关 `channel_loss_weights`，默认 `None` = 现状）
- `[1,0,0,0,0,0]`：只算 power（最激进）
- `[5,1,1,1,1,1]`：power 加权 5×（保守，**建议先试**）
- `None`：等权（现状）

代码：`src/models/configuration.py`（新增字段）+ `src/losses/criterion.py`（`DeepWindCriterion` / `DistributionCriterion` 的 channel_mask 处按权重乘 loss 与分母）。已通过数值测试 + 小模型 smoke test。

### 2.3 验证方法
- 用 `[5,1,1,1,1,1]`（或 `[1,0,0,0,0,0]`）重训 **Small** 到 100k，其余配置与 baseline Small 完全一致。
- 与 baseline Small（nCRPS 0.1097 / MAE_Cov 0.1082）比 48-cell 宏观指标。
- 判据：MAE_Coverage 明显下降 且 nCRPS 不涨。

---

## 3. Phase 1 —— 数据 / sim2real（详）

### 3.1 Step 0：数据审计（先量化，再定配方）
对比 windtoolkit 与 scada 样本的功率统计：功率 vs 风速散点、功率/可用功率比值分布、零功率占比、噪声幅度。**尚未跑**（目前是代码级推断，具体扰动幅度待审计定）。

### 3.2 Step 1：在线 sim2real 增广（不落盘）
训练加载时对 windtoolkit 的 power 通道随机注入，幅度/频率/时长对齐真实 SCADA 统计量：
1. **弃风限电**：随机时段把功率压到可用功率的 30–90%；
2. **停机事件**：随机时段功率置 ~0；
3. **量测噪声**：叠加小高斯/乘性噪声。

### 3.3 Step 2：重平衡采样
把 709 个真实风场的 53× 过采样降到 ~10–20×，减少记忆；增广后合成数据承担更多。

### 3.4 Step 3：验证
「原数据 vs 增广数据」各训一个 Small（其他一致），比 nCRPS / MAE_Coverage。

---

## 4. Phase 2 —— 统一 variate 本体

- 建立统一映射：0=功率, 1=轮毂风速, 2=风向, 3=温度, 4=气压, 5=空气密度, …；给每个源写 `channel → variate_id` 映射，重生成 metadata；106 个空 `variate_ids` 补成 `0`。
- 注意测试集 csg_wind_5 用 `0,1,2,4`（缺 3、多 4），与训练 SCADA 布局不一致，是同一坑。

---

## 5. Phase 3–5 —— 其余消融（简述）

- **Phase 3 context_length**：8192 可能过量且 8× 算力；消融 1024/2048/4096/8192，判据 2048≈8192 则降。注意 gefc12/14 已自动回退 512。
- **Phase 4 MoE→dense**：MoE 是容量杠杆非正确性杠杆；dense≈MoE 则可简化。
- **Phase 5 预测头**：quantile(21) 与 nCRPS 对齐，不是根因；对照 9/5 分位、MSE 点头、student_t。注意 `zi_beta` 在 `use_arcsinh=true` 下不可用（`criterion.py:236-241`）。

---

## 6. Phase 6 —— 规模确认 + 兜底

- 用 Phase 0/1 胜出配置重训 Base（888M）→ Large，确认规模仍提升。
- 兜底（若 Phase 0/1 都无效）：说明 decoder-only + next-patch AR 路线本身不适配，转架构重设计（非自回归解码、显式物理先验、CRPS 直接优化、弱化自监督）。

---

## 附：关键文件索引

- 目标错位：`src/models/deepwind.py:113-131`、`src/losses/criterion.py`
- 只评测 channel 0：`src/evaluation/evaluator.py:106-114`
- loss 开关（已实现）：`src/models/configuration.py`（`channel_loss_weights`）、`src/losses/criterion.py`
- variate 本体：`src/layers/embeddings.py`
- context 回退：`src/data/datasets.py:774`
- 数据采样/权重：`src/data/datasets.py:341-373`
- nCRPS：`src/utils/metrics.py:175-224`
