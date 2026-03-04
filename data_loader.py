"""
Data Loader for Pecan Street Dataset
Processes 1-minute interval data into hourly profiles for microgrid simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class PecanStreetDataLoader:
    """Load and process Pecan Street energy data for microgrid simulation."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the Pecan Street CSV file
        """
        self.csv_path = Path(csv_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[Dict[str, pd.DataFrame]] = None
        self.daily_profiles: Optional[List[Dict[str, np.ndarray]]] = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw CSV data."""
        print(f"Loading data from {self.csv_path}...")
        self.raw_data = pd.read_csv(self.csv_path)
        self.raw_data['Datetime (UTC)'] = pd.to_datetime(self.raw_data['Datetime (UTC)'])
        print(f"Loaded {len(self.raw_data):,} rows")
        print(f"Homes: {self.raw_data['Home ID'].unique()}")
        print(f"Circuits: {self.raw_data['Circuit'].unique()}")
        print(f"Date range: {self.raw_data['Datetime (UTC)'].min()} to {self.raw_data['Datetime (UTC)'].max()}")
        return self.raw_data
    
    def extract_circuit_data(self, circuit_name: str) -> pd.DataFrame:
        """
        Extract data for a specific circuit.
        
        Args:
            circuit_name: Name of the circuit (e.g., 'Solar', 'Main Panel')
            
        Returns:
            DataFrame with datetime, home_id, and power value
        """
        if self.raw_data is None:
            self.load_raw_data()
            
        circuit_data = self.raw_data[self.raw_data['Circuit'] == circuit_name].copy()
        circuit_data = circuit_data[['Datetime (UTC)', 'Home ID', 'Value']]
        circuit_data.columns = ['datetime', 'home_id', 'power_kw']
        return circuit_data
    
    def aggregate_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate 1-minute data to hourly averages.
        
        Args:
            df: DataFrame with datetime, home_id, power_kw columns
            
        Returns:
            DataFrame with hourly aggregated data
        """
        df = df.copy()
        df['hour'] = df['datetime'].dt.floor('H')
        df['date'] = df['datetime'].dt.date
        
        # Group by hour and home, calculate mean power
        hourly = df.groupby(['date', 'hour', 'home_id'])['power_kw'].mean().reset_index()
        return hourly
    
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """
        Process raw data into hourly solar and load profiles.
        
        Returns:
            Dictionary with 'solar' and 'load' DataFrames
        """
        if self.raw_data is None:
            self.load_raw_data()
            
        print("\nProcessing solar generation data...")
        solar_raw = self.extract_circuit_data('Solar')
        # Solar generation is typically negative in Pecan Street (generation = negative consumption)
        solar_raw['power_kw'] = solar_raw['power_kw'].abs()  # Make positive for generation
        solar_hourly = self.aggregate_to_hourly(solar_raw)
        
        print("Processing main panel (load) data...")
        load_raw = self.extract_circuit_data('Main Panel')
        load_hourly = self.aggregate_to_hourly(load_raw)
        
        self.processed_data = {
            'solar': solar_hourly,
            'load': load_hourly
        }
        
        print(f"\nProcessed {len(solar_hourly)} hourly solar records")
        print(f"Processed {len(load_hourly)} hourly load records")
        
        return self.processed_data
    
    def create_daily_profiles(self, home_ids: Optional[List[int]] = None) -> List[Dict[str, np.ndarray]]:
        """
        Create daily 24-hour profiles for simulation episodes.
        
        Args:
            home_ids: List of home IDs to include (None = all homes)
            
        Returns:
            List of dictionaries, each containing 24-hour solar and load arrays
        """
        if self.processed_data is None:
            self.process_data()
            
        solar_df = self.processed_data['solar']
        load_df = self.processed_data['load']
        
        if home_ids is None:
            home_ids = solar_df['home_id'].unique()
            
        daily_profiles = []
        
        for home_id in home_ids:
            home_solar = solar_df[solar_df['home_id'] == home_id]
            home_load = load_df[load_df['home_id'] == home_id]
            
            # Get unique dates for this home
            dates = home_solar['date'].unique()
            
            for date in dates:
                day_solar = home_solar[home_solar['date'] == date].sort_values('hour')
                day_load = home_load[home_load['date'] == date].sort_values('hour')
                
                # Only include complete 24-hour days
                if len(day_solar) == 24 and len(day_load) == 24:
                    solar_values = day_solar['power_kw'].values.astype(np.float32)
                    # Apply realistic solar profile - zero generation at night
                    # Night hours: 0-5 (before sunrise) and 20-23 (after sunset)
                    for h in range(24):
                        if h < 6 or h >= 20:
                            solar_values[h] = 0.0
                    
                    daily_profiles.append({
                        'home_id': home_id,
                        'date': date,
                        'solar_kw': solar_values,
                        'load_kw': day_load['power_kw'].values.astype(np.float32)
                    })
        
        self.daily_profiles = daily_profiles
        print(f"\nCreated {len(daily_profiles)} complete daily profiles")
        return daily_profiles
    
    def get_episode_data(self, episode_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get data for a single episode (day).
        
        Args:
            episode_idx: Specific episode index (None = random)
            
        Returns:
            Dictionary with 24-hour solar and load arrays
        """
        if self.daily_profiles is None:
            self.create_daily_profiles()
            
        if episode_idx is None:
            episode_idx = np.random.randint(0, len(self.daily_profiles))
        else:
            episode_idx = episode_idx % len(self.daily_profiles)
            
        return self.daily_profiles[episode_idx]
    
    def get_num_episodes(self) -> int:
        """Get the total number of available episodes."""
        if self.daily_profiles is None:
            self.create_daily_profiles()
        return len(self.daily_profiles)


class ElectricityDataLoader:
    """Load and process electricity consumption and production data."""
    
    def __init__(self, csv_path: str):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the CSV file (electricityConsumptionAndProductioction.csv)
        """
        self.csv_path = Path(csv_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.daily_profiles: Optional[List[Dict[str, np.ndarray]]] = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load the raw CSV data."""
        print(f"Loading data from {self.csv_path}...")
        self.raw_data = pd.read_csv(self.csv_path)
        self.raw_data['DateTime'] = pd.to_datetime(self.raw_data['DateTime'])
        print(f"Loaded {len(self.raw_data):,} rows")
        print(f"Date range: {self.raw_data['DateTime'].min()} to {self.raw_data['DateTime'].max()}")
        print(f"Consumption range: {self.raw_data['Consumption'].min():.0f} - {self.raw_data['Consumption'].max():.0f} W")
        print(f"Solar range: {self.raw_data['Solar'].min():.0f} - {self.raw_data['Solar'].max():.0f} W")
        return self.raw_data
    
    def process_data(self) -> pd.DataFrame:
        """Process and prepare the data."""
        if self.raw_data is None:
            self.load_raw_data()
        
        # Add date and hour columns
        self.raw_data['date'] = self.raw_data['DateTime'].dt.date
        self.raw_data['hour'] = self.raw_data['DateTime'].dt.hour
        
        # Convert W to kW
        self.raw_data['consumption_kw'] = self.raw_data['Consumption'] / 1000.0
        self.raw_data['solar_kw'] = self.raw_data['Solar'] / 1000.0
        
        print(f"Converted to kW - Consumption: {self.raw_data['consumption_kw'].mean():.2f} avg, Solar: {self.raw_data['solar_kw'].mean():.2f} avg")
        return self.raw_data
    
    def create_daily_profiles(self) -> List[Dict[str, np.ndarray]]:
        """
        Create daily 24-hour profiles for simulation episodes.
        
        Returns:
            List of dictionaries, each containing 24-hour solar and load arrays
        """
        if self.raw_data is None or 'consumption_kw' not in self.raw_data.columns:
            self.process_data()
            
        daily_profiles = []
        
        # Get unique dates
        dates = self.raw_data['date'].unique()
        
        for date in dates:
            day_data = self.raw_data[self.raw_data['date'] == date].sort_values('hour')
            
            # Only include complete 24-hour days
            if len(day_data) == 24:
                daily_profiles.append({
                    'home_id': 'aggregate',
                    'date': date,
                    'solar_kw': day_data['solar_kw'].values.astype(np.float32),
                    'load_kw': day_data['consumption_kw'].values.astype(np.float32)
                })
        
        self.daily_profiles = daily_profiles
        print(f"\nCreated {len(daily_profiles)} complete daily profiles")
        return daily_profiles
    
    def get_episode_data(self, episode_idx: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get data for a single episode (day).
        
        Args:
            episode_idx: Specific episode index (None = random)
            
        Returns:
            Dictionary with 24-hour solar and load arrays
        """
        if self.daily_profiles is None:
            self.create_daily_profiles()
            
        if episode_idx is None:
            episode_idx = np.random.randint(0, len(self.daily_profiles))
        else:
            episode_idx = episode_idx % len(self.daily_profiles)
            
        return self.daily_profiles[episode_idx]
    
    def get_num_episodes(self) -> int:
        """Get the total number of available episodes."""
        if self.daily_profiles is None:
            self.create_daily_profiles()
        return len(self.daily_profiles)


def get_tou_prices(hours: int = 24) -> np.ndarray:
    """
    Generate Time-of-Use (TOU) electricity prices.
    
    Typical TOU structure:
    - Off-peak (12am-6am): Low price
    - Mid-peak (6am-2pm, 9pm-12am): Medium price  
    - On-peak (2pm-9pm): High price
    
    Args:
        hours: Number of hours (default 24)
        
    Returns:
        Array of electricity prices ($/kWh)
    """
    prices = np.zeros(hours, dtype=np.float32)
    
    for h in range(hours):
        if 0 <= h < 6:  # Off-peak: midnight to 6am
            prices[h] = 0.08
        elif 6 <= h < 14:  # Mid-peak: 6am to 2pm
            prices[h] = 0.15
        elif 14 <= h < 21:  # On-peak: 2pm to 9pm
            prices[h] = 0.28
        else:  # Mid-peak: 9pm to midnight
            prices[h] = 0.15
            
    return prices


def get_feed_in_tariff(hours: int = 24) -> np.ndarray:
    """
    Generate feed-in tariff rates for selling excess solar.
    Typically lower than purchase prices.
    
    Args:
        hours: Number of hours (default 24)
        
    Returns:
        Array of feed-in tariff rates ($/kWh)
    """
    # Feed-in tariff is typically 30-50% of retail price
    return get_tou_prices(hours) * 0.4


if __name__ == "__main__":
    # Test the data loader
    loader = PecanStreetDataLoader("PecanStreet_10_Homes_1Min_Data.csv")
    loader.load_raw_data()
    loader.process_data()
    profiles = loader.create_daily_profiles()
    
    # Display sample profile
    if profiles:
        sample = profiles[0]
        print(f"\nSample profile - Home {sample['home_id']}, Date: {sample['date']}")
        print(f"Solar (first 6 hours): {sample['solar_kw'][:6]}")
        print(f"Load (first 6 hours): {sample['load_kw'][:6]}")
        
    # Display TOU prices
    prices = get_tou_prices()
    print(f"\nTOU Prices: {prices}")
