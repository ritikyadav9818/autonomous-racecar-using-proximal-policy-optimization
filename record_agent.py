import os
import imageio
import numpy as np
import gymnasium as gym
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecTransposeImage, VecFrameStack
)

# CONFIG 
MODEL_PATH   = "models/best_model/best_model.zip"
GIF_PATH     = "videos/gameplay.gif"
MP4_PATH     = "videos/gameplay.mp4"
SAMPLES_PATH = "results/environment_samples.png"
MAX_STEPS    = 1000
FPS          = 30
ATTEMPTS     = 20

# LOAD MODEL 
print("="*50)
print("RECORDING AGENT — PPO CarRacing-v2")
print("="*50)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print(f"Loading model: {MODEL_PATH}")

# ENVIRONMENT 
def make_env():
    def _init():
        env = gym.make(
            "CarRacing-v2",
            continuous=True,
            render_mode="rgb_array"
        )
        env = Monitor(env)
        return env
    return _init

env = DummyVecEnv([make_env()])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

model = PPO.load(MODEL_PATH, env=env)
print(f"✓ Model loaded — device: {model.device}")

# OVERLAY HELPER 
def add_overlay(frame, step, total_reward, action):
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[0], frame.shape[1]
    draw.rectangle([0, h - 30, w, h], fill=(0, 0, 0))
    text = (
        f"Step: {step:4d} | "
        f"Reward: {total_reward:8.2f} | "
        f"Steer: {action[0]:6.2f} | "
        f"Gas: {action[1]:5.2f} | "
        f"Brake: {action[2]:5.2f}"
    )
    draw.text((8, h - 22), text, fill=(255, 255, 255))
    return np.array(img)

# RECORD — multiple attempts, keep best
print(f"\nRecording gameplay — {ATTEMPTS} attempts, keeping best...")

BEST_REWARD = -999
BEST_FRAMES = []

for attempt in range(1, ATTEMPTS + 1):
    obs_try   = env.reset()
    frames_try = []
    total_try  = 0.0

    for step in range(1, MAX_STEPS + 1):
        action, _ = model.predict(obs_try, deterministic=True)
        obs_try, reward, done, info = env.step(action)
        total_try += float(reward[0])

        try:
            frame = env.venv.venv.envs[0].render()
        except Exception:
            frame = env.render()

        frame = add_overlay(frame, step, total_try, action[0])
        frames_try.append(frame)

        if done[0]:
            break

    print(
        f"  Attempt {attempt} | "
        f"Reward: {total_try:8.2f} | "
        f"Frames: {len(frames_try)}"
    )

    if total_try > BEST_REWARD:
        BEST_REWARD = total_try
        BEST_FRAMES = frames_try.copy()

frames       = BEST_FRAMES
total_reward = BEST_REWARD
print(f"\n✓ Best attempt: {BEST_REWARD:.2f} — using for GIF/MP4")

# VALIDATE 
if len(frames) == 0:
    raise RuntimeError("No frames recorded — check model path and environment")

# SAVE GIF 
os.makedirs("videos", exist_ok=True)
print(f"\nSaving GIF → {GIF_PATH}")
imageio.mimsave(GIF_PATH, frames, fps=FPS, loop=0)
print("✓ GIF saved")

# SAVE MP4 
print(f"Saving MP4 → {MP4_PATH}")
imageio.mimsave(MP4_PATH, frames, fps=FPS, quality=8, macro_block_size=1)
print("✓ MP4 saved")

# ENVIRONMENT SAMPLES 
print(f"\nCapturing environment samples → {SAMPLES_PATH}")

sample_indices = [
    max(0, int(len(frames) * 0.05)),
    max(0, int(len(frames) * 0.20)),
    max(0, int(len(frames) * 0.40)),
    max(0, int(len(frames) * 0.60)),
    max(0, int(len(frames) * 0.80)),
    len(frames) - 1
]
sample_labels = ["Start", "Early", "Corner", "Straight", "Late", "Finish"]

frame_h  = frames[0].shape[0]
frame_w  = frames[0].shape[1]
padding  = 8
label_h  = 24
canvas_w = len(sample_indices) * frame_w + (len(sample_indices) + 1) * padding
canvas_h = frame_h + label_h + padding * 2

canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
draw   = ImageDraw.Draw(canvas)

for i, (idx, label) in enumerate(zip(sample_indices, sample_labels)):
    if idx >= len(frames):
        continue
    x = padding + i * (frame_w + padding)
    canvas.paste(Image.fromarray(frames[idx]), (x, label_h))
    draw.text((x + 10, 4), label, fill=(255, 255, 255))

os.makedirs("results", exist_ok=True)
canvas.save(SAMPLES_PATH)
print("✓ Environment samples saved")

# SUMMARY 
print("\n" + "="*50)
print("RECORDING COMPLETE")
print("="*50)
print(f"  GIF      : {GIF_PATH}")
print(f"  MP4      : {MP4_PATH}")
print(f"  Samples  : {SAMPLES_PATH}")
print(f"  Reward   : {total_reward:.2f}")
print(f"  Frames   : {len(frames)}")
print(f"  Attempts : {ATTEMPTS}")
print("="*50)

env.close()