import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Log file path
# reward_file = '/home/yu/git/IR-BPP/logs/runs/IR-blockoutexphier-2025.07.20-14-50-35/events.out.tfevents.1753015837.yu-ASUS-TUF-Gaming-A15-FA507XI-FA507XI'
reward_file = '/home/yu/git/IR-BPP/result/IR-blockoutexphier-2025.07.01-23-56-04/events.out.tfevents.1751406966.yu-ASUS-TUF-Gaming-A15-FA507XI-FA507XI'
# Load reward log
reward_ea = event_accumulator.EventAccumulator(reward_file)
reward_ea.Reload()

# Extract reward data
reward_events = reward_ea.Scalars('Metric/Reward_mean')
steps = np.array([e.step for e in reward_events])
rewards = np.array([e.value for e in reward_events])

# Smoothed reward (moving average)
def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_rewards = moving_average(rewards)
smoothed_steps = steps[:len(smoothed_rewards)]

# Calculate the upper and lower fluctuation range
std = np.std(rewards) * 0.1
upper = smoothed_rewards + std
lower = smoothed_rewards - std

# Show
plt.figure(figsize=(10, 6))
plt.plot(smoothed_steps, smoothed_rewards, label='Smoothed Reward', color='tab:blue', linewidth=2)
plt.fill_between(smoothed_steps, lower, upper, color='tab:blue', alpha=0.2, label='± std dev')
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.title('Training Reward Curve', fontsize=16)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
