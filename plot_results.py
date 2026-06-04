import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.summary.summary_iterator import summary_iterator

# CONFIG 
TB_DIR       = "logs/tensorboard/PPO_2"
EVAL_DIR     = "logs/eval"
SUMMARY_PATH = "results/training_summary.json"
RESULTS_DIR  = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# STYLE 
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "axes.titlecolor":   "#C9D1D9",
    "text.color":        "#C9D1D9",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "grid.color":        "#21262D",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#161B22",
    "legend.edgecolor":  "#30363D",
    "lines.linewidth":   2.0,
    "font.family":       "DejaVu Sans",
})

CYAN    = "#00B4D8"
GREEN   = "#3FB950"
ORANGE  = "#F0883E"
RED     = "#F85149"
PURPLE  = "#BC8CFF"
YELLOW  = "#E3B341"

# READ TENSORBOARD 
def read_tb(tb_dir, tags):
    """Read scalar values from TensorBoard event files."""
    data = {tag: {"steps": [], "values": []} for tag in tags}

    event_files = [
        os.path.join(tb_dir, f)
        for f in os.listdir(tb_dir)
        if f.startswith("events.out.tfevents")
    ]

    for ef in sorted(event_files):
        try:
            for event in summary_iterator(ef):
                step = event.step
                for v in event.summary.value:
                    if v.tag in data:
                        data[v.tag]["steps"].append(step)
                        data[v.tag]["values"].append(v.simple_value)
        except Exception as e:
            print(f"  Warning reading {ef}: {e}")

    # Sort by step
    for tag in data:
        pairs = sorted(zip(data[tag]["steps"], data[tag]["values"]))
        if pairs:
            data[tag]["steps"], data[tag]["values"] = zip(*pairs)
            data[tag]["steps"]  = list(data[tag]["steps"])
            data[tag]["values"] = list(data[tag]["values"])

    return data

# READ EVAL RESULTS 
def read_eval(eval_dir):
    """Read EvalCallback evaluations.npz if it exists."""
    npz_path = os.path.join(eval_dir, "evaluations.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    return {
        "timesteps": data["timesteps"],
        "results":   data["results"],    # shape (n_evals, n_episodes)
        "ep_lengths": data.get("ep_lengths", None)
    }

# LOAD SUMMARY JSON 
summary = {}
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)
    print(f"✓ Loaded training_summary.json")

milestones = summary.get("training", {}).get("milestones", [])

# READ TENSORBOARD DATA
print(f"Reading TensorBoard logs from {TB_DIR}...")

TAGS = [
    "rollout/ep_rew_mean",
    "train/entropy_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/loss",
    "time/total_timesteps",
]

tb = read_tb(TB_DIR, TAGS)

rew_steps  = tb["rollout/ep_rew_mean"]["steps"]
rew_values = tb["rollout/ep_rew_mean"]["values"]
ent_steps  = tb["train/entropy_loss"]["steps"]
ent_values = tb["train/entropy_loss"]["values"]
ev_steps   = tb["train/explained_variance"]["steps"]
ev_values  = tb["train/explained_variance"]["values"]
lr_steps   = tb["train/learning_rate"]["steps"]
lr_values  = tb["train/learning_rate"]["values"]
loss_steps = tb["train/loss"]["steps"]
loss_values= tb["train/loss"]["values"]

print(f"  rollout/ep_rew_mean    : {len(rew_values)} points")
print(f"  train/entropy_loss     : {len(ent_values)} points")
print(f"  train/explained_variance: {len(ev_values)} points")
print(f"  train/learning_rate    : {len(lr_values)} points")
print(f"  train/loss             : {len(loss_values)} points")

eval_data = read_eval(EVAL_DIR)
if eval_data is not None:
    print(f"  eval data              : {len(eval_data['timesteps'])} evals")
else:
    print(f"  eval data              : not found")

# SMOOTH HELPER 
def smooth(values, weight=0.85):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


# PLOT 1 — REWARD CURVE

print("\nGenerating reward_curve.png...")

fig, ax = plt.subplots(figsize=(12, 5))

if rew_values:
    ax.plot(rew_steps, rew_values,
            color=CYAN, alpha=0.25, linewidth=1, label="_raw")
    ax.plot(rew_steps, smooth(rew_values),
            color=CYAN, linewidth=2.5, label="Mean Episode Reward (smoothed)")

# Milestone markers from JSON
if milestones:
    m_steps  = [m["timestep"]    for m in milestones]
    m_rewards= [m["mean_reward"] for m in milestones]
    ax.scatter(m_steps, m_rewards, color=YELLOW, s=60,
               zorder=5, label="Milestone (Mean-20)")

# Best eval line
best_ts = summary.get("evaluation", {}).get("best_timestep")
if best_ts:
    ax.axvline(best_ts, color=GREEN, linestyle="--",
               alpha=0.7, label=f"Best model saved ({best_ts:,} steps)")

ax.set_title("Training Reward Curve", fontsize=14, pad=12)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Episode Reward")
ax.legend(loc="upper left")
ax.grid(True)

# Annotate peak
if rew_values:
    peak_idx = int(np.argmax(smooth(rew_values)))
    ax.annotate(
        f"Peak: {smooth(rew_values)[peak_idx]:.0f}",
        xy=(rew_steps[peak_idx], smooth(rew_values)[peak_idx]),
        xytext=(rew_steps[peak_idx], smooth(rew_values)[peak_idx] + 50),
        color=YELLOW, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1)
    )

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/reward_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ reward_curve.png")


# PLOT 2 — EVALUATION REWARDS

print("Generating evaluation_rewards.png...")

fig, ax = plt.subplots(figsize=(12, 5))

if eval_data is not None:
    ts       = eval_data["timesteps"]
    means    = eval_data["results"].mean(axis=1)
    stds     = eval_data["results"].std(axis=1)

    ax.fill_between(ts, means - stds, means + stds,
                    color=GREEN, alpha=0.15, label="±1 std")
    ax.plot(ts, means, color=GREEN, linewidth=2.5,
            label="Mean Eval Reward (5 episodes)")
    ax.scatter(ts, means, color=GREEN, s=25, zorder=4)

    # Best eval point
    best_idx = int(np.argmax(means))
    ax.scatter(ts[best_idx], means[best_idx],
               color=YELLOW, s=100, zorder=5,
               label=f"Best eval: {means[best_idx]:.1f} @ {ts[best_idx]:,}")

elif milestones:
    # Fallback — use milestone data
    m_steps   = [m["timestep"]    for m in milestones]
    m_rewards = [m["mean_reward"] for m in milestones]
    ax.plot(m_steps, m_rewards, color=GREEN, linewidth=2.5,
            marker="o", markersize=6, label="Milestone Mean(20)")

ax.set_title("Evaluation Rewards Over Training", fontsize=14, pad=12)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Mean Eval Reward")
ax.legend(loc="upper left")
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/evaluation_rewards.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ evaluation_rewards.png")


# PLOT 3 — ENTROPY CURVE

print("Generating entropy_curve.png...")

fig, ax = plt.subplots(figsize=(12, 5))

if ent_values:
    ax.plot(ent_steps, ent_values,
            color=PURPLE, alpha=0.25, linewidth=1)
    ax.plot(ent_steps, smooth(ent_values),
            color=PURPLE, linewidth=2.5, label="Policy Entropy Loss")

ax.set_title("Policy Entropy Over Training", fontsize=14, pad=12)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Entropy Loss")
ax.legend(loc="upper right")
ax.grid(True)
ax.invert_yaxis()  # entropy loss is negative — invert for readability

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/entropy_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ entropy_curve.png")


# PLOT 4 — EXPLAINED VARIANCE

print("Generating explained_variance.png...")

fig, ax = plt.subplots(figsize=(12, 5))

if ev_values:
    ax.plot(ev_steps, ev_values,
            color=ORANGE, alpha=0.25, linewidth=1)
    ax.plot(ev_steps, smooth(ev_values),
            color=ORANGE, linewidth=2.5, label="Explained Variance")
    ax.axhline(1.0, color="#8B949E", linestyle="--",
               alpha=0.5, label="Perfect = 1.0")
    ax.axhline(0.0, color=RED, linestyle="--",
               alpha=0.5, label="Random = 0.0")

ax.set_title("Explained Variance Over Training", fontsize=14, pad=12)
ax.set_xlabel("Timesteps")
ax.set_ylabel("Explained Variance")
ax.set_ylim(-1.5, 1.2)
ax.legend(loc="lower right")
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/explained_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ explained_variance.png")


# PLOT 5 — TRAINING TIME / THROUGHPUT

print("Generating training_time.png...")

fig, ax = plt.subplots(figsize=(12, 5))

# Use milestone data for throughput proxy
if milestones and len(milestones) > 1:
    m_steps = [0] + [m["timestep"] for m in milestones]
    # Steps per interval
    intervals    = [m_steps[i+1] - m_steps[i] for i in range(len(m_steps)-1)]
    interval_mid = [(m_steps[i+1] + m_steps[i]) / 2 for i in range(len(m_steps)-1)]
    m_rewards    = [m["mean_reward"] for m in milestones]

    ax2 = ax.twinx()
    ax.bar(interval_mid, intervals, width=150_000,
           color=CYAN, alpha=0.3, label="Steps per interval")
    ax2.plot(
        [m["timestep"] for m in milestones],
        m_rewards,
        color=YELLOW, linewidth=2.5, marker="o",
        markersize=6, label="Mean(20) reward at milestone"
    )
    ax2.set_ylabel("Mean Reward", color=YELLOW)
    ax2.tick_params(axis="y", labelcolor=YELLOW)
    ax2.spines["right"].set_edgecolor(YELLOW)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Steps per Interval", color=CYAN)
    ax.tick_params(axis="y", labelcolor=CYAN)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax.set_title("Training Progress — Steps vs Reward at Milestones",
             fontsize=14, pad=12)
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/training_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ training_time.png")


# SUMMARY

print("\n" + "="*50)
print("ALL PLOTS GENERATED")
print("="*50)
plots = [
    "reward_curve.png",
    "evaluation_rewards.png",
    "entropy_curve.png",
    "explained_variance.png",
    "training_time.png",
]
for p in plots:
    path   = os.path.join(RESULTS_DIR, p)
    exists = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {exists}  {p}")
print("="*50)