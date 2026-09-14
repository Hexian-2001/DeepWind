# tools/debug_rolling.py
import numpy as np
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
NPY_PATH     = "/scratch/pawsey0115/hwang4/deepwindData/test/csg_wind_5.npy"
RESULTS_PATH = "/scratch/pawsey0115/hwang4/results/DeepWind-Research/results/deepwind/deepwind_small_fix_sharding_v2/raw_results/csg_wind_5/raw_H12.npz"
OUTPUT_DIR   = "/scratch/pawsey0115/hwang4/results/DeepWind-Research/results/deepwind/deepwind_small_fix_sharding_v2"

VARIATE_IDX      = 0
CONTEXT_LENGTH   = 8192
PRED_LEN         = 48
TRAIN_RATIO      = 0.7
VAL_RATIO        = 0.1
N_STEPS          = 1440   # steps to compare

# ── Load npy ──────────────────────────────────────────────────────────────────
data = np.load(NPY_PATH)          # (C, T)
print(f"npy shape: {data.shape}")

T         = data.shape[1]
t_train   = int(T * TRAIN_RATIO)
t_val     = int(T * (TRAIN_RATIO + VAL_RATIO))
# test split starts at t_val
test_start = t_val

# Raw ground truth from npy (same slice as targets in evaluation)
raw = data[VARIATE_IDX, test_start + CONTEXT_LENGTH :
                         test_start + CONTEXT_LENGTH + N_STEPS]

print(f"test split start : {test_start}")
print(f"raw slice        : [{test_start + CONTEXT_LENGTH}, "
      f"{test_start + CONTEXT_LENGTH + N_STEPS})")
print(f"raw shape        : {raw.shape}")
    
# ── Load results ──────────────────────────────────────────────────────────────
res              = np.load(RESULTS_PATH)
targets_full     = res["targets"].reshape(-1)      # (N * pred_len,)
point_preds_full = res["point_preds"].reshape(-1)

print(f"\ntargets_full shape     : {targets_full.shape}")
print(f"point_preds_full shape : {point_preds_full.shape}")

# ── Numerical comparison ──────────────────────────────────────────────────────
n = min(N_STEPS, len(raw), len(targets_full))

print(f"\n── Numerical check (first {n} steps) ──")
print(f"raw          min={raw[:n].min():.4f}  max={raw[:n].max():.4f}  mean={raw[:n].mean():.4f}")
print(f"targets_full min={targets_full[:n].min():.4f}  max={targets_full[:n].max():.4f}  mean={targets_full[:n].mean():.4f}")
print(f"max abs diff = {np.abs(raw[:n] - targets_full[:n]).max():.6f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
t = np.arange(n)

fig, axes = plt.subplots(3, 1, figsize=(20, 9), sharex=True)

# Panel 1: raw vs targets (should be identical if no bug)
axes[0].plot(t, raw[:n],          label="raw (npy)",     color="black",     lw=1)
axes[0].plot(t, targets_full[:n], label="targets (npz)", color="tomato",    lw=1, ls="--")
axes[0].set_title("Raw npy vs concatenated targets — should overlap perfectly")
axes[0].legend()

# Panel 2: forecast vs ground truth
axes[1].plot(t, targets_full[:n],     label="ground truth", color="black",     lw=1)
axes[1].plot(t, point_preds_full[:n], label="forecast",     color="steelblue", lw=1, alpha=0.85)
# Mark horizon boundaries
for b in range(0, n, PRED_LEN):
    axes[1].axvline(b, color="gray", lw=0.4, ls="--", alpha=0.5)
axes[1].set_title("Rolling forecast vs ground truth")
axes[1].legend()

# Panel 3: residual (raw - targets)
axes[2].plot(t, raw[:n] - targets_full[:n], color="purple", lw=0.8)
axes[2].axhline(0, color="black", lw=0.5)
axes[2].set_title("Residual: raw - targets (should be ~0 everywhere)")
axes[2].set_xlabel("Time step")

plt.tight_layout()

out_path = f"{OUTPUT_DIR}/debug_rolling.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")