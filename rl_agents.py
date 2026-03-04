"""
Reinforcement Learning Agents for Microgrid Energy Management
Implements SAC, PPO, and Robust-SAC using Stable-Baselines3.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, Callable, List
from pathlib import Path

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


class TrainingCallback(BaseCallback):
    """
    Custom callback for tracking training progress.
    """
    
    def __init__(self, log_interval: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log progress
        if self.n_calls % self.log_interval == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Step {self.n_calls}: Mean reward (last 10): {mean_reward:.2f}")
        return True
    
    def _on_rollout_end(self) -> None:
        # Collect episode stats from infos
        if hasattr(self, 'locals') and 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])


def create_sac_agent(
    env: gym.Env,
    learning_rate: float = 3e-4,
    buffer_size: int = 50000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_kwargs: Optional[Dict] = None,
    verbose: int = 1
) -> SAC:
    """
    Create a SAC (Soft Actor-Critic) agent.
    
    Args:
        env: The gymnasium environment
        learning_rate: Learning rate for all networks
        buffer_size: Size of the replay buffer
        batch_size: Minibatch size for training
        gamma: Discount factor
        tau: Soft update coefficient
        policy_kwargs: Additional policy network arguments
        verbose: Verbosity level
        
    Returns:
        Configured SAC agent
    """
    if policy_kwargs is None:
        policy_kwargs = {
            'net_arch': [256, 256]
        }
    
    agent = SAC(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=None  # Can be enabled for TensorBoard logging
    )
    
    return agent


def create_ppo_agent(
    env: gym.Env,
    learning_rate: float = 3e-4,    # Lower LR for stability
    n_steps: int = 4096,            # Larger rollout for better gradient estimates
    batch_size: int = 128,          # Smaller batches = more updates per rollout
    n_epochs: int = 15,             # Slightly fewer epochs to prevent overfitting
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,        # Tighter clipping for stable updates
    ent_coef: float = 0.01,         # Higher entropy for better exploration
    vf_coef: float = 0.5,           # Value function coefficient
    max_grad_norm: float = 0.5,     # Gradient clipping for stability
    policy_kwargs: Optional[Dict] = None,
    verbose: int = 1
) -> PPO:
    """
    Create a PPO (Proximal Policy Optimization) agent.
    
    Args:
        env: The gymnasium environment
        learning_rate: Learning rate
        n_steps: Number of steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        clip_range: PPO clip range
        ent_coef: Entropy coefficient for exploration
        vf_coef: Value function loss coefficient
        max_grad_norm: Max gradient norm for clipping
        policy_kwargs: Additional policy network arguments
        verbose: Verbosity level
        
    Returns:
        Configured PPO agent
    """
    if policy_kwargs is None:
        policy_kwargs = {
            'net_arch': dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Deeper network
            'ortho_init': True,  # Orthogonal initialization for better training
        }
    
    agent = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=True,              # Enable State Dependent Exploration
        sde_sample_freq=4,         # Sample noise every 4 steps
        normalize_advantage=True,  # Normalize advantages for stable training
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=None
    )
    
    return agent


def create_robust_sac_agent(
    env: gym.Env,
    learning_rate: float = 1e-4,     # Lower LR for stable robust learning
    buffer_size: int = 100000,       # Larger buffer for diverse experience
    batch_size: int = 512,           # Larger batch for stable gradients
    gamma: float = 0.99,
    tau: float = 0.005,
    train_freq: int = 1,             # Update every step
    gradient_steps: int = 2,         # More gradient steps per update
    learning_starts: int = 1000,     # Warmup steps before learning
    policy_kwargs: Optional[Dict] = None,
    verbose: int = 1,
    **sac_kwargs
) -> tuple:
    """
    Create a Robust SAC agent with domain randomization.
    
    Uses enhanced hyperparameters for better robustness:
    - Lower learning rate for stable convergence under noise
    - Larger replay buffer to retain diverse experiences
    - Larger batch size for more stable gradient estimates
    - Deeper network architecture for complex pattern recognition
    
    Args:
        env: The base gymnasium environment (should have randomization enabled)
        learning_rate: Learning rate (lower for stability)
        buffer_size: Size of replay buffer
        batch_size: Minibatch size
        gamma: Discount factor
        tau: Soft update coefficient
        train_freq: Training frequency
        gradient_steps: Gradient steps per update
        learning_starts: Steps before learning starts
        policy_kwargs: Network architecture configuration
        verbose: Verbosity level
        **sac_kwargs: Additional arguments for SAC
        
    Returns:
        (agent, env) tuple
    """
    if policy_kwargs is None:
        policy_kwargs = {
            'net_arch': [256, 256, 128],  # Deeper network for robust learning
        }
    
    # Create robust SAC agent with tuned hyperparameters
    agent = SAC(
        policy='MlpPolicy',
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        learning_starts=learning_starts,
        use_sde=True,                    # State-dependent exploration for robustness
        use_sde_at_warmup=True,          # Use SDE during warmup too
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        tensorboard_log=None,
        **sac_kwargs
    )
    
    return agent, env


def train_agent(
    agent,
    total_timesteps: int,
    callback: Optional[BaseCallback] = None,
    progress_bar: bool = True
) -> Dict[str, Any]:
    """
    Train an RL agent.
    
    Args:
        agent: The RL agent (SAC or PPO)
        total_timesteps: Total training steps
        callback: Optional training callback
        progress_bar: Show progress bar
        
    Returns:
        Training statistics
    """
    print(f"\n{'='*50}")
    print(f"Training {type(agent).__name__} for {total_timesteps:,} timesteps...")
    print(f"{'='*50}\n")
    
    if callback is None:
        callback = TrainingCallback(log_interval=1000)
    
    agent.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=progress_bar
    )
    
    return {
        'timesteps': total_timesteps,
        'episode_rewards': getattr(callback, 'episode_rewards', []),
        'episode_lengths': getattr(callback, 'episode_lengths', [])
    }


def save_agent(agent, path: str):
    """Save agent to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    agent.save(path)
    print(f"Agent saved to {path}")


def load_agent(agent_class, path: str, env: gym.Env):
    """Load agent from file."""
    return agent_class.load(path, env=env)


def evaluate_agent(
    agent,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: Trained RL agent
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy
        
    Returns:
        Evaluation statistics
    """
    all_rewards = []
    all_costs = []
    all_episodes = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        all_rewards.append(episode_reward)
        all_costs.append(info['total_cost'])
        all_episodes.append(env.get_episode_results())
    
    return {
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'mean_cost': np.mean(all_costs),
        'std_cost': np.std(all_costs),
        'all_rewards': all_rewards,
        'all_costs': all_costs,
        'episodes': all_episodes
    }


class NoopAgent:
    """Baseline agent that takes no action (battery idle)."""
    
    def predict(self, obs, deterministic=True):
        return np.array([0.0]), None


class RandomAgent:
    """Baseline agent that takes random actions."""
    
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None


class RuleBasedAgent:
    """
    Rule-based baseline agent:
    - Charge when solar > load and price is low
    - Discharge when solar < load and price is high
    """
    
    def predict(self, obs, deterministic=True):
        # obs: [soc, net_power, price, hour, prev_action, history...]
        soc = (obs[0] + 1) / 2  # Convert [-1,1] to [0,1]
        net_power = obs[1]  # Positive = need power
        price = obs[2]  # Normalized price
        
        action = 0.0
        
        if net_power < -0.2:  # Excess solar
            if soc < 0.9:
                action = 0.8  # Charge
        elif net_power > 0.2:  # Need power
            if soc > 0.2:
                if price > 0.1:  # High price
                    action = -0.8  # Discharge
                    
        return np.array([action]), None


if __name__ == "__main__":
    # Test agent creation
    from microgrid_env import MicrogridEnv
    from data_loader import get_tou_prices
    
    # Create test environment
    np.random.seed(42)
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    env = MicrogridEnv(solar, load, prices)
    
    # Test SAC creation
    print("Creating SAC agent...")
    sac_agent = create_sac_agent(env, verbose=0)
    print(f"SAC agent created: {sac_agent}")
    
    # Test PPO creation
    print("\nCreating PPO agent...")
    ppo_agent = create_ppo_agent(env, verbose=0)
    print(f"PPO agent created: {ppo_agent}")
    
    # Test Robust SAC creation
    print("\nCreating Robust SAC agent...")
    rsac_agent, rsac_env = create_robust_sac_agent(env, verbose=0)
    print(f"Robust SAC agent created: {rsac_agent}")
    
    # Quick training test
    print("\nQuick training test (100 steps)...")
    train_agent(sac_agent, 100, progress_bar=False)
    print("Training test passed!")
