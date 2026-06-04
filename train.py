import os
import json
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecTransposeImage, VecFrameStack
)

# CONFIG 
SEED            = 42
TOTAL_TIMESTEPS = 2_000_000
CHECKPOINT_FREQ = 200_000
EVAL_FREQ       = 50_000
N_EVAL_EPISODES = 5
N_ENVS          = 4

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

HYPERPARAMS = {
    "learning_rate": linear_schedule(3e-4),
    "n_steps":       1024,
    "batch_size":    64,
    "n_epochs":      10,
    "gamma":         0.99,
    "gae_lambda":    0.95,
    "clip_range":    0.2,
    "vf_coef":       0.5,
    "max_grad_norm": 0.5,
}

# ENVIRONMENT FACTORY 
def make_env(seed=0):
    def _init():
        env = gym.make("CarRacing-v2", continuous=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

# TRAINING SUMMARY CALLBACK 
class TrainingSummaryCallback(BaseCallback):
    def __init__(
        self,
        save_path="results/training_summary.json",
        verbose=0
    ):
        super().__init__(verbose)
        self.save_path       = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.summary = {
            "environment":            "CarRacing-v2",
            "algorithm":              "PPO",
            "policy":                 "CnnPolicy",
            "seed":                   SEED,
            "n_envs":                 N_ENVS,
            "hyperparameters": {
                "learning_rate": "linear_schedule(3e-4 → 0)",
                "n_steps":       1024,
                "batch_size":    64,
                "n_epochs":      10,
                "gamma":         0.99,
                "gae_lambda":    0.95,
                "clip_range":    0.2,
                "vf_coef":       0.5,
                "max_grad_norm": 0.5,
                "frame_stack":   4
            },
            "total_timesteps_target": TOTAL_TIMESTEPS,
            "training": {
                "final_timesteps":          None,
                "total_episodes":           None,
                "best_episode_reward":      None,
                "worst_episode_reward":     None,
                "mean_episode_reward":      None,
                "final_mean_reward_last20": None,
                "milestones":               []
            },
            "evaluation": {
                "best_mean_reward": None,
                "best_timestep":    None,
                "n_eval_episodes":  N_EVAL_EPISODES,
                "eval_frequency":   EVAL_FREQ
            }
        }

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)

                n = len(self.episode_rewards)
                if n % 10 == 0:
                    mean20 = np.mean(self.episode_rewards[-20:])
                    print(
                        f"  Ep {n:>5} | "
                        f"Reward: {r:>8.2f} | "
                        f"Mean(20): {mean20:>8.2f} | "
                        f"Steps: {self.num_timesteps:>9,}"
                    )
        return True

    def _on_training_end(self):
        t  = self.summary["training"]
        rs = self.episode_rewards
        t["final_timesteps"]  = self.num_timesteps
        t["total_episodes"]   = len(rs)
        if rs:
            t["best_episode_reward"]      = round(max(rs), 2)
            t["worst_episode_reward"]     = round(min(rs), 2)
            t["mean_episode_reward"]      = round(float(np.mean(rs)), 2)
            t["final_mean_reward_last20"] = round(
                float(np.mean(rs[-min(20, len(rs)):])), 2
            )
        self._save()
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print(f"  Total episodes  : {t['total_episodes']}")
        print(f"  Best reward     : {t['best_episode_reward']}")
        print(f"  Mean reward     : {t['mean_episode_reward']}")
        print(f"  Final mean(20)  : {t['final_mean_reward_last20']}")
        print(f"  Summary saved   : {self.save_path}")
        print("="*50 + "\n")

    def add_milestone(self, timestep, mean_reward):
        self.summary["training"]["milestones"].append({
            "timestep":    timestep,
            "mean_reward": round(float(mean_reward), 2)
        })
        self._save()

    def _save(self):
        with open(self.save_path, "w") as f:
            json.dump(self.summary, f, indent=2)


# TRACKING EVAL CALLBACK
class TrackingEvalCallback(EvalCallback):
    """
    EvalCallback that writes best_mean_reward and
    best_timestep to JSON only when a NEW best model
    is discovered.
    """
    def __init__(self, summary_cb, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_cb = summary_cb

    def _on_step(self):
        result = super()._on_step()

        if self.best_mean_reward != -np.inf:
            current_best = round(float(self.best_mean_reward), 2)
            stored_best  = self.summary_cb.summary[
                "evaluation"
            ]["best_mean_reward"]

            if stored_best is None or current_best > stored_best:
                self.summary_cb.summary["evaluation"][
                    "best_mean_reward"
                ] = current_best
                self.summary_cb.summary["evaluation"][
                    "best_timestep"
                ] = self.num_timesteps
                self.summary_cb._save()

        return result


# MILESTONE CALLBACK 
class MilestoneCallback(BaseCallback):
    def __init__(self, summary_cb, interval=CHECKPOINT_FREQ, verbose=0):
        super().__init__(verbose)
        self.summary_cb     = summary_cb
        self.interval       = interval
        self.next_milestone = interval

    def _on_step(self):
        if self.num_timesteps >= self.next_milestone:
            rs = self.summary_cb.episode_rewards
            if rs:
                mean20 = float(np.mean(rs[-20:]))
                self.summary_cb.add_milestone(
                    self.num_timesteps, mean20
                )
                print(
                    f"\n── Milestone {self.next_milestone:,} │ "
                    f"Mean(20): {mean20:.2f} ──\n"
                )
            self.next_milestone += self.interval
        return True


#  MAIN — required on Windows for SubprocVecEnv 
if __name__ == "__main__":

    # PATHS
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("models/best_model",  exist_ok=True)
    os.makedirs("logs/tensorboard",   exist_ok=True)
    os.makedirs("logs/eval",          exist_ok=True)
    os.makedirs("results",            exist_ok=True)
    os.makedirs("videos",             exist_ok=True)
    os.makedirs("notebooks",          exist_ok=True)

    # DEVICE
    device_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    print(f"Training on: {device_name}")

    # ENVIRONMENTS 
    env = SubprocVecEnv([
        make_env(seed=SEED + i) for i in range(N_ENVS)
    ])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_env(seed=SEED + 99)])
    eval_env = VecTransposeImage(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)

    print(f"Environments ready — CarRacing-v2 x{N_ENVS} parallel")

    # CALLBACKS 
    summary_cb = TrainingSummaryCallback(
        save_path="results/training_summary.json"
    )
    milestone_cb = MilestoneCallback(
        summary_cb=summary_cb,
        interval=CHECKPOINT_FREQ
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ // N_ENVS,
        save_path="./models/checkpoints/",
        name_prefix="ppo_carracing",
        verbose=1
    )
    eval_cb = TrackingEvalCallback(
        summary_cb=summary_cb,
        eval_env=eval_env,
        best_model_save_path="./models/best_model/",
        log_path="./logs/eval/",
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1
    )

    # MODEL 
    model = PPO(
        "CnnPolicy",
        env,
        **HYPERPARAMS,
        seed=SEED,
        tensorboard_log="./logs/tensorboard/",
        verbose=0
    )

    # PRINT SUMMARY 
    print("\n" + "="*50)
    print("AUTONOMOUS RACE CAR — PPO TRAINING v2")
    print("="*50)
    print(f"  Device      : {device_name}")
    print(f"  Environment : CarRacing-v2 x{N_ENVS} parallel")
    print(f"  Frame stack : 4 frames")
    print(f"  LR schedule : 3e-4 → 0 (linear decay)")
    print(f"  n_steps     : 1024")
    print(f"  Seed        : {SEED}")
    print(f"  Timesteps   : {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoints : every {CHECKPOINT_FREQ:,} steps")
    print(f"  Evaluation  : every {EVAL_FREQ:,} steps")
    print(f"  Est. time   : several hours (hardware dependent)")
    print("="*50 + "\n")

    # TRAIN
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            checkpoint_cb,
            eval_cb,
            summary_cb,
            milestone_cb
        ],
        progress_bar=True
    )

    # SAVE FINAL 
    model.save("models/ppo_carracing_final")
    print("Final model → models/ppo_carracing_final.zip")

    env.close()
    eval_env.close()
