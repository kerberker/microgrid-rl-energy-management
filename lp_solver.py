"""
Linear Programming (LP) Solver for Optimal Battery Scheduling
Provides theoretical lower bound for microgrid energy management.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.optimize import linprog, minimize
from dataclasses import dataclass


@dataclass
class LPConfig:
    """Configuration for LP optimization."""
    e_max: float = 13.5  # Maximum battery capacity (kWh)
    e_min_ratio: float = 0.1  # Minimum SOC ratio
    p_bat_max: float = 5.0  # Maximum charge/discharge power (kW)
    eta_charge: float = 0.95  # Charging efficiency
    eta_discharge: float = 0.95  # Discharging efficiency
    ramp_rate: float = 2.5  # Maximum ramp rate (kW/step)
    p_grid_peak: float = 10.0  # Peak demand threshold
    peak_penalty_rate: float = 0.50  # Peak penalty ($/kW)
    degradation_cost: float = 0.02  # Degradation cost ($/kWh)
    feed_in_ratio: float = 0.4  # Feed-in tariff as fraction of retail
    
    @property
    def e_min(self) -> float:
        return self.e_max * self.e_min_ratio


class MicrogridLPSolver:
    """
    Solve the optimal battery scheduling problem using linear programming.
    
    The optimization minimizes:
        sum_t (energy_cost[t] + peak_penalty[t] + degradation[t])
    
    Subject to:
        - SOC dynamics: SOC[t+1] = SOC[t] + η_c*P_c[t] - P_d[t]/η_d
        - SOC bounds: E_min <= SOC[t] <= E_max
        - Power bounds: 0 <= P_c[t] <= P_bat_max, 0 <= P_d[t] <= P_bat_max
        - Ramping: |P_bat[t] - P_bat[t-1]| <= ramp_rate
        - Grid power: P_grid[t] = Load[t] - Solar[t] + P_c[t] - P_d[t]
    """
    
    def __init__(self, config: Optional[LPConfig] = None):
        self.config = config or LPConfig()
        
    def solve(
        self,
        solar_profile: np.ndarray,
        load_profile: np.ndarray,
        price_profile: np.ndarray,
        initial_soc: float = 0.5,
        final_soc_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """
        Solve the optimal battery scheduling problem.
        
        Args:
            solar_profile: 24-hour solar generation (kW)
            load_profile: 24-hour load consumption (kW)
            price_profile: 24-hour electricity prices ($/kWh)
            initial_soc: Initial state of charge [0, 1]
            final_soc_penalty: Penalty for deviating from initial SOC at end
            
        Returns:
            Dictionary with optimal schedule and costs
        """
        T = len(solar_profile)
        net_load = load_profile - solar_profile  # Positive = need power
        
        # Decision variables:
        # P_c[0:T] - charging power (T variables)
        # P_d[T:2T] - discharging power (T variables)
        # P_peak[2T:3T] - peak violation (T variables)
        # P_grid_pos[3T:4T] - positive grid power (buying)
        # P_grid_neg[4T:5T] - negative grid power (selling)
        
        n_vars = 5 * T
        
        # Objective: minimize energy cost + peak penalty + degradation
        c = np.zeros(n_vars)
        
        for t in range(T):
            # Energy cost for buying
            c[3*T + t] = price_profile[t]
            # Revenue for selling (negative cost)
            c[4*T + t] = -price_profile[t] * self.config.feed_in_ratio
            # Peak penalty
            c[2*T + t] = self.config.peak_penalty_rate
            # Degradation cost (on both charge and discharge)
            c[t] += self.config.degradation_cost * self.config.eta_charge
            c[T + t] += self.config.degradation_cost / self.config.eta_discharge
        
        # Build constraint matrices
        A_eq = []
        b_eq = []
        A_ub = []
        b_ub = []
        
        # SOC dynamics constraints (equality)
        soc_current = initial_soc * self.config.e_max
        
        for t in range(T):
            # SOC[t+1] = SOC[t] + eta_c * P_c[t] - P_d[t] / eta_d
            # We track cumulative SOC
            pass
        
        # Grid power balance: P_grid_pos - P_grid_neg = net_load + P_c - P_d
        for t in range(T):
            row = np.zeros(n_vars)
            row[3*T + t] = 1  # P_grid_pos
            row[4*T + t] = -1  # -P_grid_neg
            row[t] = -1  # -P_c
            row[T + t] = 1  # +P_d
            A_eq.append(row)
            b_eq.append(net_load[t])
        
        # Peak violation: P_peak >= P_grid_pos - P_grid_peak
        for t in range(T):
            row = np.zeros(n_vars)
            row[2*T + t] = -1  # -P_peak
            row[3*T + t] = 1  # P_grid_pos
            A_ub.append(row)
            b_ub.append(self.config.p_grid_peak)
        
        # SOC bounds (cumulative)
        for t in range(T):
            # Upper bound: SOC_init + sum(eta_c*P_c - P_d/eta_d) <= E_max
            row_upper = np.zeros(n_vars)
            for s in range(t + 1):
                row_upper[s] = self.config.eta_charge  # P_c
                row_upper[T + s] = -1 / self.config.eta_discharge  # P_d
            A_ub.append(row_upper)
            b_ub.append(self.config.e_max - soc_current)
            
            # Lower bound: SOC_init + sum(eta_c*P_c - P_d/eta_d) >= E_min
            # Rewrite as: -sum(...) <= E_min - SOC_init
            row_lower = np.zeros(n_vars)
            for s in range(t + 1):
                row_lower[s] = -self.config.eta_charge
                row_lower[T + s] = 1 / self.config.eta_discharge
            A_ub.append(row_lower)
            b_ub.append(soc_current - self.config.e_min)
        
        # Final SOC constraint: end at 50% SOC (with small tolerance)
        # SOC_init + sum(eta_c*P_c - P_d/eta_d) = 0.5 * E_max
        target_soc = 0.5 * self.config.e_max
        row_final = np.zeros(n_vars)
        for s in range(T):
            row_final[s] = self.config.eta_charge
            row_final[T + s] = -1 / self.config.eta_discharge
        A_eq.append(row_final)
        b_eq.append(target_soc - soc_current)
        
        # Ramping constraints
        for t in range(1, T):
            # |P_bat[t] - P_bat[t-1]| <= ramp_rate
            # P_bat = P_c - P_d
            # (P_c[t] - P_d[t]) - (P_c[t-1] - P_d[t-1]) <= ramp_rate
            row1 = np.zeros(n_vars)
            row1[t] = 1  # P_c[t]
            row1[T + t] = -1  # -P_d[t]
            row1[t-1] = -1  # -P_c[t-1]
            row1[T + t-1] = 1  # P_d[t-1]
            A_ub.append(row1)
            b_ub.append(self.config.ramp_rate)
            
            # -(P_c[t] - P_d[t]) + (P_c[t-1] - P_d[t-1]) <= ramp_rate
            row2 = np.zeros(n_vars)
            row2[t] = -1
            row2[T + t] = 1
            row2[t-1] = 1
            row2[T + t-1] = -1
            A_ub.append(row2)
            b_ub.append(self.config.ramp_rate)
        
        # Variable bounds
        bounds = []
        # P_c bounds
        for t in range(T):
            bounds.append((0, self.config.p_bat_max))
        # P_d bounds
        for t in range(T):
            bounds.append((0, self.config.p_bat_max))
        # P_peak bounds (non-negative)
        for t in range(T):
            bounds.append((0, None))
        # P_grid_pos bounds (non-negative)
        for t in range(T):
            bounds.append((0, None))
        # P_grid_neg bounds (non-negative)
        for t in range(T):
            bounds.append((0, None))
        
        # Convert to arrays
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        
        # Solve LP
        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'
        )
        
        if not result.success:
            print(f"LP solver warning: {result.message}")
            # Return fallback solution
            return self._get_fallback_solution(solar_profile, load_profile, price_profile, initial_soc)
        
        # Extract solution
        x = result.x
        P_c = x[0:T]
        P_d = x[T:2*T]
        P_peak = x[2*T:3*T]
        P_grid_pos = x[3*T:4*T]
        P_grid_neg = x[4*T:5*T]
        
        # Calculate SOC trajectory
        soc_trajectory = [initial_soc]
        for t in range(T):
            delta_soc = (P_c[t] * self.config.eta_charge - P_d[t] / self.config.eta_discharge) / self.config.e_max
            soc_trajectory.append(np.clip(soc_trajectory[-1] + delta_soc, self.config.e_min_ratio, 1.0))
        soc_trajectory = np.array(soc_trajectory[1:])
        
        # Calculate costs
        P_grid = P_grid_pos - P_grid_neg
        P_bat = P_c - P_d
        
        energy_cost = np.sum(P_grid_pos * price_profile - P_grid_neg * price_profile * self.config.feed_in_ratio)
        peak_penalty = np.sum(P_peak * self.config.peak_penalty_rate)
        throughput = np.sum(np.abs(P_c * self.config.eta_charge) + np.abs(P_d / self.config.eta_discharge)) / 2
        degradation_cost = throughput * self.config.degradation_cost
        total_cost = energy_cost + peak_penalty + degradation_cost
        
        return {
            'success': result.success,
            'total_cost': total_cost,
            'energy_cost': energy_cost,
            'peak_penalty': peak_penalty,
            'degradation_cost': degradation_cost,
            'battery_power': P_bat,
            'grid_power': P_grid,
            'soc_trajectory': soc_trajectory,
            'charging_power': P_c,
            'discharging_power': P_d,
            'peak_violations': P_peak,
            'throughput': throughput,
            'final_soc': soc_trajectory[-1],
            'cumulative_cost': np.cumsum(
                P_grid_pos * price_profile - P_grid_neg * price_profile * self.config.feed_in_ratio +
                P_peak * self.config.peak_penalty_rate
            ),
            'solar_profile': solar_profile,
            'load_profile': load_profile,
            'price_profile': price_profile
        }
    
    def _get_fallback_solution(
        self,
        solar_profile: np.ndarray,
        load_profile: np.ndarray,
        price_profile: np.ndarray,
        initial_soc: float
    ) -> Dict[str, Any]:
        """
        Get a simple fallback solution when LP fails.
        Uses a greedy heuristic.
        """
        T = len(solar_profile)
        net_load = load_profile - solar_profile
        
        soc = initial_soc
        soc_trajectory = []
        battery_power = []
        grid_power = []
        costs = []
        
        for t in range(T):
            # Simple heuristic: charge when excess solar, discharge when needed and price is high
            if net_load[t] < 0 and soc < 0.9:  # Excess solar
                p_bat = min(-net_load[t], self.config.p_bat_max, (1 - soc) * self.config.e_max)
            elif net_load[t] > 0 and soc > 0.2 and price_profile[t] > 0.15:  # High price, need power
                p_bat = -min(net_load[t], self.config.p_bat_max, (soc - self.config.e_min_ratio) * self.config.e_max)
            else:
                p_bat = 0
            
            # Update SOC
            if p_bat > 0:
                delta_soc = p_bat * self.config.eta_charge / self.config.e_max
            else:
                delta_soc = p_bat / self.config.eta_discharge / self.config.e_max
            soc = np.clip(soc + delta_soc, self.config.e_min_ratio, 1.0)
            
            # Calculate grid power and cost
            p_grid = net_load[t] + p_bat
            if p_grid >= 0:
                cost = p_grid * price_profile[t]
            else:
                cost = p_grid * price_profile[t] * self.config.feed_in_ratio
            cost += max(0, p_grid - self.config.p_grid_peak) * self.config.peak_penalty_rate
            
            soc_trajectory.append(soc)
            battery_power.append(p_bat)
            grid_power.append(p_grid)
            costs.append(cost)
        
        return {
            'success': False,
            'total_cost': sum(costs),
            'energy_cost': sum(c for c, g in zip(costs, grid_power) if g >= 0),
            'peak_penalty': sum(max(0, g - self.config.p_grid_peak) * self.config.peak_penalty_rate for g in grid_power),
            'degradation_cost': 0,
            'battery_power': np.array(battery_power),
            'grid_power': np.array(grid_power),
            'soc_trajectory': np.array(soc_trajectory),
            'charging_power': np.maximum(battery_power, 0),
            'discharging_power': np.maximum(-np.array(battery_power), 0),
            'peak_violations': np.maximum(np.array(grid_power) - self.config.p_grid_peak, 0),
            'throughput': sum(abs(p) for p in battery_power),
            'final_soc': soc,
            'cumulative_cost': np.cumsum(costs),
            'solar_profile': solar_profile,
            'load_profile': load_profile,
            'price_profile': price_profile
        }


def solve_lp_benchmark(
    solar_profile: np.ndarray,
    load_profile: np.ndarray,
    price_profile: np.ndarray,
    initial_soc: float = 0.5,
    config: Optional[LPConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to solve LP benchmark.
    
    Args:
        solar_profile: 24-hour solar generation
        load_profile: 24-hour load consumption
        price_profile: 24-hour prices
        initial_soc: Initial SOC
        config: LP configuration
        
    Returns:
        LP solution dictionary
    """
    solver = MicrogridLPSolver(config)
    return solver.solve(solar_profile, load_profile, price_profile, initial_soc)


if __name__ == "__main__":
    # Test the LP solver
    from data_loader import get_tou_prices
    
    np.random.seed(42)
    
    # Create test profiles
    solar = np.maximum(0, 5 * np.sin(np.linspace(0, np.pi, 24)))
    load = 3 + 2 * np.sin(np.linspace(0, 2*np.pi, 24) + np.pi/4)
    prices = get_tou_prices()
    
    print("Testing LP Solver...")
    print(f"Solar profile: {solar[:6].round(2)}...")
    print(f"Load profile: {load[:6].round(2)}...")
    print(f"Price profile: {prices[:6]}...")
    
    # Solve
    result = solve_lp_benchmark(solar, load, prices, initial_soc=0.5)
    
    print(f"\nLP Solution:")
    print(f"  Success: {result['success']}")
    print(f"  Total cost: ${result['total_cost']:.2f}")
    print(f"  Energy cost: ${result['energy_cost']:.2f}")
    print(f"  Peak penalty: ${result['peak_penalty']:.2f}")
    print(f"  Final SOC: {result['final_soc']:.2%}")
    print(f"  Throughput: {result['throughput']:.2f} kWh")
