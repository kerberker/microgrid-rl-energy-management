"""
Microgrid Environment for Reinforcement Learning
Custom Gymnasium environment for battery energy management.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

class ScenarioGenerator:
    """Generate perturbed scenarios with forecast errors."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def add_noise(
        self,
        profile: np.ndarray,
        noise_level: float,
        correlation: float = 0.8, # Default high correlation for persistence
        clip_min: float = 0.0,
        clip_max: float = None
    ) -> np.ndarray:
        """
        Add Correlated Gaussian noise (Autoregressive AR(1) process).
        noise_t = rho * noise_{t-1} + sqrt(1 - rho^2) * epsilon_t
        """
        n_steps = len(profile)
        noise = np.zeros(n_steps)
        
        if abs(correlation) < 1e-5:
            # Revert to white noise if correlation is 0
            noise = self.rng.normal(0, noise_level, n_steps)
        else:
            # AR(1) Process
            # Initial sample
            noise[0] = self.rng.normal(0, noise_level)
            
            # Scale for the innovation term to maintain constant variance = noise_level^2
            innovation_scale = noise_level * np.sqrt(1 - correlation**2)
            
            for t in range(1, n_steps):
                noise[t] = correlation * noise[t-1] + self.rng.normal(0, innovation_scale)
            
        noisy_profile = profile * (1 + noise)
        
        if clip_max is None:
            clip_max = profile.max() * 2
        
        return np.clip(noisy_profile, clip_min, clip_max).astype(np.float32)
    
    def generate_scenarios(
        self,
        solar_profile: np.ndarray,
        load_profile: np.ndarray,
        price_profile: np.ndarray,
        noise_level: float,
        n_scenarios: int,
        correlation: float = 0.8
    ) -> List[Dict[str, np.ndarray]]:
        """Generate multiple noisy scenarios."""
        scenarios = []
        
        for _ in range(n_scenarios):
            scenario = {
                'solar': self.add_noise(solar_profile, noise_level, correlation, clip_min=0.0),
                'load': self.add_noise(load_profile, noise_level, correlation, clip_min=0.1),
                'price': self.add_noise(price_profile, noise_level * 0.5, correlation, clip_min=0.01),
                'noise_level': noise_level
            }
            scenarios.append(scenario)
        
        return scenarios



@dataclass
class MicrogridConfig:
    """Configuration parameters for the microgrid environment."""
    # Battery parameters
    e_max: float = 13.5  # Maximum battery capacity (kWh) - Tesla Powerwall
    e_min_ratio: float = 0.1  # Minimum SOC ratio (10%)
    p_bat_max: float = 5.0  # Maximum charge/discharge power (kW)
    eta_charge: float = 0.95  # Charging efficiency
    eta_discharge: float = 0.95  # Discharging efficiency
    ramp_rate: float = 2.5  # Maximum power ramp rate (kW/step)
    
    # Grid parameters
    p_grid_peak: float = 10.0  # Peak demand threshold (kW)
    peak_penalty_rate: float = 0.50  # $/kW over threshold
    
    # Battery degradation (simplified)
    degradation_cost_per_kwh: float = 0.02  # $/kWh throughput
    
    # Terminal SOC constraint
    target_final_soc: float = 0.5  # Target SOC at end of episode (50%)
    terminal_soc_penalty: float = 5.0  # $/per 10% deviation from target
    
    # Simulation parameters
    hours_per_day: int = 24
    history_length: int = 4  # Number of historical steps to include in state
    forecast_horizon: int = 24  # Number of future steps to include in state (Look-ahead window)

    
    @property
    def e_min(self) -> float:
        return self.e_max * self.e_min_ratio


class MicrogridEnv(gym.Env):
    """
    Custom Gymnasium environment for microgrid energy management.
    
    The agent controls battery charging/discharging to minimize total cost
    (energy cost + peak penalty + degradation) over a 24-hour period.
    
    State Space includes a look-ahead window (forecast) of PV, Load, and Price
    for the next 24 hours (t:t+24) to allow for energy management planning.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        solar_profile: np.ndarray,
        load_profile: np.ndarray,
        price_profile: np.ndarray,
        actual_solar_profile: Optional[np.ndarray] = None,
        actual_load_profile: Optional[np.ndarray] = None,
        actual_price_profile: Optional[np.ndarray] = None,
        config: Optional[MicrogridConfig] = None,
        render_mode: Optional[str] = None,

        randomize_env: bool = False,
        noise_level: float = 0.1,
        variable_noise_level: bool = False,
        correlation: float = 0.8, # Default persistence
        seed: int = 42
    ):
        """
        Initialize the microgrid environment.
        
        Args:
            solar_profile: Forecast solar generation profile (kW) - What agent sees
            load_profile: Forecast load consumption profile (kW) - What agent sees
            price_profile: Forecast electricity price profile ($/kWh)
            actual_*: Actual realization profiles (used for step physics). If None, equals forecast.
            config: Environment configuration
            render_mode: Rendering mode ('human' or 'ansi')
            randomize_env: Whether to randomize ACTUAL profiles on reset (Training Mode)
            noise_level: Noise level for randomization (if enabled)
            seed: Random seed
        """
        super().__init__()
        
        self.config = config or MicrogridConfig()
        self.render_mode = render_mode
        self.randomize_env = randomize_env
        self.noise_level = noise_level
        self.variable_noise_level = variable_noise_level
        self.correlation = correlation
        self.current_noise_level = noise_level # Track current episode noise
        self.scenario_generator = ScenarioGenerator(seed=seed) if randomize_env else None

        
        # Store Forecast profiles (Observation)
        self.solar_profile = np.array(solar_profile, dtype=np.float32)
        self.load_profile = np.array(load_profile, dtype=np.float32)
        self.price_profile = np.array(price_profile, dtype=np.float32)
        
        # Store Actual profiles (Physics) - Default to forecast if not provided
        self.actual_solar_profile = np.array(actual_solar_profile if actual_solar_profile is not None else solar_profile, dtype=np.float32)
        self.actual_load_profile = np.array(actual_load_profile if actual_load_profile is not None else load_profile, dtype=np.float32)
        self.actual_price_profile = np.array(actual_price_profile if actual_price_profile is not None else price_profile, dtype=np.float32)
        
        assert len(self.solar_profile) == self.config.hours_per_day
        assert len(self.load_profile) == self.config.hours_per_day
        assert len(self.price_profile) == self.config.hours_per_day
        
        # Calculate normalization factors
        self.max_net_power = max(abs(self.load_profile - self.solar_profile).max(), 1.0)
        self.max_price = max(self.price_profile.max(), 0.01)
        
        # Define observation space
        # [SOC, net_power, price, sin_hour, cos_hour, prev_action, last_error, history..., forecast...]
        # Forecast includes solar, load, price, sin_hour, cos_hour for each future step
        base_obs_dim = 7 # sin/cos hour (2) + SOC, net, price, prev_action, last_error (5) -> 7
        history_dim = self.config.history_length
        forecast_dim = self.config.forecast_horizon * 5  # Solar, Load, Price, Sin, Cos per step
        
        obs_dim = base_obs_dim + history_dim + forecast_dim
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Define action space: battery power [-P_BAT_MAX, P_BAT_MAX]
        # Negative = discharge, Positive = charge
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self._reset_state()
        
    def _reset_state(self):
        """Reset internal state variables."""
        self.current_step = 0
        self.soc = 0.5  # Initial SOC at 50%
        self.prev_battery_power = 0.0
        self.total_cost = 0.0
        self.total_energy_cost = 0.0
        self.total_peak_penalty = 0.0
        self.total_degradation_cost = 0.0
        self.total_terminal_soc_penalty = 0.0
        self.total_throughput = 0.0
        self.total_throughput = 0.0
        self.peak_violations = 0
        self.last_forecast_error = 0.0
        self.history = np.zeros(self.config.history_length, dtype=np.float32)
        
        # Episode tracking
        self.soc_history = []
        self.battery_power_history = []
        self.grid_power_history = []
        self.cost_history = []
        
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        hour = self.current_step
        
        # Get current values
        solar = self.solar_profile[hour] if hour < self.config.hours_per_day else 0
        load = self.load_profile[hour] if hour < self.config.hours_per_day else 0
        price = self.price_profile[hour] if hour < self.config.hours_per_day else 0
        net_power = load - solar
        
        # Cyclical Time Encoding
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        
        # --- Get Forecasts ---
        forecast_solar_src = self.solar_profile
        forecast_load_src = self.load_profile
        forecast_price_src = self.price_profile
            
        forecast_features = []
        for i in range(1, self.config.forecast_horizon + 1):
            target_hour = (hour + i) % 24
            
            f_solar = forecast_solar_src[target_hour]
            f_load = forecast_load_src[target_hour]
            f_price = forecast_price_src[target_hour]
            f_net = f_load - f_solar
            
            # Cyclical encoding for forecast step
            f_sin = np.sin(2 * np.pi * target_hour / 24.0)
            f_cos = np.cos(2 * np.pi * target_hour / 24.0)
            
            forecast_features.extend([
                np.clip(f_net / self.max_net_power, -1, 1),
                f_price / self.max_price - 0.5,
                f_solar / self.max_net_power, # Normalized Solar
                f_sin,
                f_cos
            ])

        # Construct Observation Vector
        obs = np.array([
            (self.soc - 0.5) * 2,  # SOC: [0,1] -> [-1,1]
            np.clip(net_power / self.max_net_power, -1, 1),  # Net power
            price / self.max_price - 0.5,  # Price
            sin_hour,
            cos_hour,
            self.prev_battery_power / self.config.p_bat_max,  # Previous action
            np.clip(self.last_forecast_error / self.max_net_power, -1, 1), # Last forecast error
            *self.history,  # Historical net power
            *forecast_features # Forecast window
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            'step': self.current_step,
            'soc': self.soc,
            'total_cost': self.total_cost,
            'energy_cost': self.total_energy_cost,
            'peak_penalty': self.total_peak_penalty,
            'degradation_cost': self.total_degradation_cost,
            'throughput': self.total_throughput,
            'peak_violations': self.peak_violations
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., 'initial_soc')
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Apply domain randomization if enabled
        # Apply domain randomization if enabled (Training Mode)
        if self.randomize_env:
            # Determine noise level for this episode
            if self.variable_noise_level:
                self.current_noise_level = self.np_random.uniform(0.0, self.noise_level)
            else:
                self.current_noise_level = self.noise_level

            self.actual_solar_profile = self.scenario_generator.add_noise(self.solar_profile, self.current_noise_level, self.correlation, clip_min=0.0)
            self.actual_load_profile = self.scenario_generator.add_noise(self.load_profile, self.current_noise_level, self.correlation, clip_min=0.1)
            # We typically don't randomize price as much, but let's add a little bit
            self.actual_price_profile = self.scenario_generator.add_noise(self.price_profile, self.current_noise_level * 0.2, self.correlation, clip_min=0.01)
            
            # Recalculate normalization based on FORECAST (Agent shouldn't know the noise scale perfectly)
            self.max_net_power = max(abs(self.load_profile - self.solar_profile).max(), 1.0)
        
        # If explicit actuals were not provided in init and we are not randomizing, reset actuals to forecast
        elif not self.randomize_env:
             # In evaluation mode where we might want to manually set actuals, 
             # we assume the user has set them or they remain what they were initialized as.
             # However, to be safe for standard resets:
             if hasattr(self, 'initial_actual_solar'):
                 # Reset to original actuals if we stored them (would need to implement this storage)
                 pass
             pass # For now, assume if not randomizing, the profiles set in __init__ (or reset args) are static
        
        self._reset_state()
        
        # Set initial SOC (random or specified)
        if options and 'initial_soc' in options:
            self.soc = np.clip(options['initial_soc'], self.config.e_min_ratio, 1.0)
        else:
            # Random initial SOC between 20% and 80%
            self.soc = self.np_random.uniform(0.2, 0.8)
            
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Battery power action [-1, 1], scaled to [-P_BAT_MAX, P_BAT_MAX]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Scale action to actual power
        action_value = float(action[0]) if hasattr(action, '__len__') else float(action)
        battery_power_requested = np.clip(action_value, -1, 1) * self.config.p_bat_max
        
        # Apply ramping constraint
        power_change = battery_power_requested - self.prev_battery_power
        if abs(power_change) > self.config.ramp_rate:
            power_change = np.sign(power_change) * self.config.ramp_rate
        battery_power = self.prev_battery_power + power_change
        
        # Get current profiles from ACTUAL source (The Physics of the world)
        hour = self.current_step
        solar = self.actual_solar_profile[hour]
        load = self.actual_load_profile[hour]
        price = self.actual_price_profile[hour]
        net_demand = load - solar  # Positive = need power, Negative = excess generation
        
        # Apply battery constraints
        current_energy = self.soc * self.config.e_max
        
        if battery_power > 0:  # Charging
            # Limit by remaining capacity
            max_charge_energy = (1.0 - self.soc) * self.config.e_max
            max_charge_power = max_charge_energy / 1.0  # 1 hour timestep
            battery_power = min(battery_power, max_charge_power, self.config.p_bat_max)
            # Account for charging efficiency
            energy_to_battery = battery_power * self.config.eta_charge
        else:  # Discharging
            # Limit by available energy (above minimum)
            available_energy = (self.soc - self.config.e_min_ratio) * self.config.e_max
            max_discharge_power = available_energy / 1.0  # 1 hour timestep
            battery_power = max(battery_power, -max_discharge_power, -self.config.p_bat_max)
            # Account for discharging efficiency
            energy_to_battery = battery_power / self.config.eta_discharge
        
        # Update SOC
        new_soc = self.soc + energy_to_battery / self.config.e_max
        new_soc = np.clip(new_soc, self.config.e_min_ratio, 1.0)
        
        # Calculate actual energy change
        actual_energy_change = (new_soc - self.soc) * self.config.e_max
        
        # Calculate grid power
        # Grid power = load - solar + battery charging (or - battery discharging)
        grid_power = net_demand + battery_power  # Positive = buying from grid
        
        # Calculate costs
        if grid_power >= 0:
            energy_cost = grid_power * price  # Buying from grid
        else:
            # Selling to grid (typically at lower rate)
            feed_in_tariff = price * 0.4  # 40% of retail price
            energy_cost = grid_power * feed_in_tariff  # Negative cost = revenue
        
        # Peak penalty
        peak_violation = max(0, grid_power - self.config.p_grid_peak)
        peak_penalty = peak_violation * self.config.peak_penalty_rate
        if peak_violation > 0:
            self.peak_violations += 1
        
        # Degradation cost
        throughput = abs(actual_energy_change)
        degradation_cost = throughput * self.config.degradation_cost_per_kwh
        
        # Total cost
        step_cost = energy_cost + peak_penalty + degradation_cost
        
        # Update state
        self.soc = new_soc
        self.prev_battery_power = battery_power
        self.current_step += 1
        
        # Calculate forecast error for NEXT observation
        # Error = Actual Net - Forecast Net
        # We want to know how wrong the forecast was for the step we just took (or upcoming?)
        # Convention: The observation at step t includes error from step t-1.
        # So we calculate the error of the *current* step (which just finished) to be shown in next obs.
        
        # Forecast at this step was:
        f_solar = self.solar_profile[hour]
        f_load = self.load_profile[hour]
        f_net = f_load - f_solar
        
        actual_net = net_demand # load - solar
        
        self.last_forecast_error = actual_net - f_net
        
        # Update history
        self.history = np.roll(self.history, -1)
        self.history[-1] = np.clip(net_demand / self.max_net_power, -1, 1)
        
        # Update totals
        self.total_cost += step_cost
        self.total_energy_cost += energy_cost
        self.total_peak_penalty += peak_penalty
        self.total_degradation_cost += degradation_cost
        self.total_throughput += throughput
        
        # Store history for analysis
        self.soc_history.append(self.soc)
        self.battery_power_history.append(battery_power)
        self.grid_power_history.append(grid_power)
        self.cost_history.append(step_cost)
        
        # Check termination
        terminated = self.current_step >= self.config.hours_per_day
        truncated = False
        
        # Reward is negative cost (to minimize cost)
        reward = -step_cost
        
        # Apply terminal SOC penalty at end of episode
        if terminated:
            soc_deviation = abs(self.soc - self.config.target_final_soc)
            terminal_penalty = soc_deviation * self.config.terminal_soc_penalty * 10  # Scale to make it significant
            self.total_terminal_soc_penalty = terminal_penalty
            self.total_cost += terminal_penalty
            reward -= terminal_penalty
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render the current state."""
        if self.render_mode == "human" or self.render_mode == "ansi":
            hour = self.current_step - 1 if self.current_step > 0 else 0
            print(f"Hour {hour:2d}: SOC={self.soc:.2%}, "
                  f"Grid={self.grid_power_history[-1] if self.grid_power_history else 0:.2f}kW, "
                  f"Bat={self.battery_power_history[-1] if self.battery_power_history else 0:.2f}kW, "
                  f"Cost=${self.total_cost:.2f}")
    
    def get_episode_results(self) -> Dict[str, Any]:
        """Get comprehensive results for the episode."""
        return {
            'soc_trajectory': np.array(self.soc_history),
            'battery_power': np.array(self.battery_power_history),
            'grid_power': np.array(self.grid_power_history),
            'cost_per_step': np.array(self.cost_history),
            'cumulative_cost': np.cumsum(self.cost_history),
            'total_cost': self.total_cost,
            'energy_cost': self.total_energy_cost,
            'peak_penalty': self.total_peak_penalty,
            'degradation_cost': self.total_degradation_cost,
            'throughput': self.total_throughput,
            'peak_violations': self.peak_violations,
            'final_soc': self.soc,
            'final_soc': self.soc,
            'solar_profile': self.actual_solar_profile,   # Return actuals for analysis
            'load_profile': self.actual_load_profile,
            'price_profile': self.actual_price_profile,
            'forecast_solar': self.solar_profile,         # Return forecast too
            'forecast_load': self.load_profile
        }


class DynamicMicrogridEnv(MicrogridEnv):
    """
    Microgrid environment with dynamic profile loading.
    Automatically samples new profiles each episode from a data loader.
    """
    
    def __init__(
        self,
        data_loader,
        price_profile: np.ndarray,
        config: Optional[MicrogridConfig] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize with a data loader for dynamic profile sampling.
        
        Args:
            data_loader: PecanStreetDataLoader instance
            price_profile: 24-hour electricity price profile
            config: Environment configuration
            render_mode: Rendering mode
        """
        self.data_loader = data_loader
        self.stored_price_profile = price_profile
        
        # Get initial profile
        episode_data = data_loader.get_episode_data(0)
        
        super().__init__(
            solar_profile=episode_data['solar_kw'],
            load_profile=episode_data['load_kw'],
            price_profile=price_profile,
            config=config,
            render_mode=render_mode
        )
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and load a new profile."""
        super().reset(seed=seed)
        
        # Sample new profile
        episode_data = self.data_loader.get_episode_data()
        self.solar_profile = episode_data['solar_kw']
        self.load_profile = episode_data['load_kw']
        
        # Recalculate normalization
        self.max_net_power = max(abs(self.load_profile - self.solar_profile).max(), 1.0)
        
        self._reset_state()
        
        # Set initial SOC
        if options and 'initial_soc' in options:
            self.soc = np.clip(options['initial_soc'], self.config.e_min_ratio, 1.0)
        else:
            self.soc = self.np_random.uniform(0.2, 0.8)
            
        return self._get_obs(), self._get_info()


if __name__ == "__main__":
    # Test the environment
    from data_loader import get_tou_prices
    
    # Create sample profiles
    np.random.seed(42)
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))  # Peak at noon
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)  # Typical load pattern
    prices = get_tou_prices()
    
    # Create environment
    env = MicrogridEnv(solar, load, prices, render_mode="human")
    
    # Run one episode with random actions
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial SOC: {info['soc']:.2%}")
    
    total_reward = 0
    for step in range(24):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated:
            break
    
    print(f"\nEpisode finished:")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Total cost: ${info['total_cost']:.2f}")
    print(f"Final SOC: {info['soc']:.2%}")
    print(f"Peak violations: {info['peak_violations']}")
