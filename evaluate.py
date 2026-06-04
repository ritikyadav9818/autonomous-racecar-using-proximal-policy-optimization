import os
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecFrameStack
)

# CONFIG 
MODEL_PATH   = "models/best_model/best_model.zip"
SUMMARY_PATH = "results/training_summary.json"
N_EPISODES   = 20
SEED         = 100

# LOAD 
print("="*50)
print("EVALUATION — PPO CarRacing-v2")
print("="*50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"Loading model: {MODEL_PATH}")

# ENVIRONMENT 
def make_env(seed=0):
    def _init():
        env = gym.make("CarRacing-v2", continuous=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

eval_env = DummyVecEnv([make_env(seed=SEED)])
eval_env = VecTransposeImage(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)

model = PPO.load(MODEL_PATH, env=eval_env)
print(f"✓ Model loaded — device: {model.device}")
print(f"Running {N_EPISODES} evaluation episodes...\n")

# EVALUATION LOOP 
episode_rewards = []
episode_lengths = []

for ep in range(1, N_EPISODES + 1):
    obs          = eval_env.reset()
    total_reward = 0.0
    steps        = 0
    done         = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_vec, info = eval_env.step(action)
        total_reward += float(reward[0])
        steps        += 1
        done          = bool(done_vec[0])

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    print(
        f"  Ep {ep:>3} | "
        f"Reward: {total_reward:>8.2f} | "
        f"Steps: {steps:>5} | "
        f"Seed: {SEED + ep}" 
    )

eval_env.close()

# STATISTICS 
mean_r   = float(np.mean(episode_rewards))
std_r    = float(np.std(episode_rewards))
min_r    = float(np.min(episode_rewards))
max_r    = float(np.max(episode_rewards))
median_r = float(np.median(episode_rewards))
mean_len = float(np.mean(episode_lengths))

print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
print(f"  Episodes     : {N_EPISODES}")
print(f"  Mean reward  : {mean_r:.2f}")
print(f"  Std reward   : {std_r:.2f}")
print(f"  Min reward   : {min_r:.2f}")
print(f"  Max reward   : {max_r:.2f}")
print(f"  Median       : {median_r:.2f}")
print(f"  Mean length  : {mean_len:.1f} steps")
print("="*50)

# UPDATE JSON 
if os.path.exists(SUMMARY_PATH):
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    evaluation = summary.get("evaluation", {})
    evaluation.update({
        "n_eval_episodes":    N_EPISODES,
        "mean_reward":        round(mean_r,   2),
        "std_reward":         round(std_r,    2),
        "min_reward":         round(min_r,    2),
        "max_reward":         round(max_r,    2),
        "median_reward":      round(median_r, 2),
        "mean_episode_length": round(mean_len, 1),
        "model_evaluated":    MODEL_PATH,
        "eval_seed":          SEED,
    })
    summary["evaluation"] = evaluation

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ training_summary.json updated")
else:
    print(f"\n⚠ {SUMMARY_PATH} not found — JSON not updated")