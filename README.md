Autonomous Race Car using PPO (CarRacing-v2)

This project presents a complete end-to-end Reinforcement Learning (RL) solution for the CarRacing-v2 environment using Proximal Policy Optimization (PPO). The agent learns to drive from raw pixel observations without any human-engineered rules, relying solely on deep neural networks and trial-and-error learning. Training was performed in Google Colab using a GPU-accelerated PPO pipeline, and multiple checkpoints (300k, 552k, 700k, 800k, and 900k timesteps) were saved.
The final 900k model is provided in the release section as the main deliverable.


Features

PPO agent trained on CarRacing-v2
CNN-based encoder for pixel observations
Stable-Baselines3 (v2.3.2) implementation
Multiple checkpoints saved (300k–900k steps)
Reward graph showing improvement over time
Policy network weight visualization
Evaluation script (rendered rollout or video frames)
Easy to run Google Colab notebook
Lightweight, no custom environment required

Project Structure
autonomous racecar using ppo
│
├── 📄 PPO_RaceCar_Final.ipynb      # Colab notebook (training + testing)
├── 📄 ppo_carracing_900k.zip       # Final trained model
├── 📄 reward_graph.png             # Reward trend graph
├── 📄 README.md                    # Project documentation
│
└── releases/                       # Extra model checkpoints (optional)
       ├── ppo_300k.zip
       ├── ppo_700k.zip
       ├── ppo_800k.zip
       └── ppo_900k.zip

🔧 Installation

Create a clean Python environment (recommended: Python 3.10):

pip install stable-baselines3==2.3.2
pip install gymnasium[box2d]==0.29.1
pip install pygame==2.5.2


If using Google Colab, GPU (T4) is recommended.

Usage
Load the trained model:
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CarRacing-v2", continuous=True)
model = PPO.load("ppo_carracing_900k.zip", env=env)

Run the agent:
obs, _ = env.reset()
total_reward = 0

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()

print("Episode reward:", total_reward)
env.close()

Training Details
Algorithm:

PPO (clip range: 0.1, entropy bonus, GAE)

Stable-Baselines3 2.3.2

Hyperparameters:

Learning rate: 3e-4 → reduced to 1e-4

Max KL limit: 0.07 (for stable updates)

Timesteps trained: 900,000

Frame stack: Yes (default 4 frames)

Action space: Continuous (steering, throttle, brake)

Training Hardware:

Google Colab

NVIDIA T4 GPU

The full 900k steps took ~2.5 hours

Results & Visualizations
Reward Trend Graph

Shows how the model improves from near-zero reward to stable driving behavior.

Policy Network Visualization

A plot showing the first-layer weight activations in the policy network.

Both images are included in the repo.

Release Information
Latest Release:

v1.0 – PPO CarRacing 900k Model

Includes:

ppo_carracing_900k.zip (final agent)

reward graph + weight visualization

training notebook

License

This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.
