import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, price_history, sentiment_scores, initial_balance=1000):
        super(TradingEnv, self).__init__()
        
        self.price_history = price_history
        self.sentiment_scores = sentiment_scores
        self.initial_balance = initial_balance
        
        self.n_steps = len(price_history)
        self.current_step = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [price, sentiment, balance, holding]
        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holding = 0
        self.total_value = self.balance
        return self._get_obs()

    def _get_obs(self):
        # Clamp current_step to valid range in case it's called after done
        step = min(self.current_step, self.n_steps - 1)
        price = float(self.price_history[step])
        sentiment = float(self.sentiment_scores[step])
        return np.array([price, sentiment, float(self.balance), float(self.holding)], dtype=np.float32)

    def step(self, action):
        done = False

        if self.current_step >= self.n_steps:
            done = True
            return self._get_obs(), 0.0, done, {}

        price = self.price_history[self.current_step]
        reward = 0.0

        # Action logic
        if action == 1 and self.balance >= price:
            self.holding += 1
            self.balance -= price
        elif action == 2 and self.holding > 0:
            self.holding -= 1
            self.balance += price

        self.total_value = self.balance + self.holding * price

        self.current_step += 1
        if self.current_step >= self.n_steps:
            done = True

        reward = self.total_value - self.initial_balance

        return self._get_obs(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holding}, Value: {self.total_value}")



        """
✅ What this does:

Simulates trading 1 share per action

Tracks balance, holdings, and total value

Gives reward based on portfolio gain

Uses price + sentiment for input
        """