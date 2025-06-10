import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data.data_loader import get_paired_data
from agent.env import TradingEnv

def train_agent(ticker="AAPL", total_timesteps=10000):
    # Load data
    prices, sentiments = get_paired_data(ticker)

    # Create environment
    def make_env():
        return TradingEnv(prices, sentiments)

    env = DummyVecEnv([make_env])  # Vectorized for Stable-Baselines

    # Initialize PPO agent
    model = PPO("MlpPolicy", env, verbose=1)

    # Train agent
    model.learn(total_timesteps=total_timesteps)

    # Save model
    model.save(f"ppo_trading_{ticker.lower()}")

    print(f"✅ Training complete. Model saved as ppo_trading_{ticker.lower()}")
    return model

if __name__ == "__main__":
    train_agent()



"""✅ What This Does:
Loads stock + sentiment data

Creates your custom Gym environment

Trains the PPO agent for 10,000 timesteps

Saves the model to disk
        """