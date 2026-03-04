
import numpy as np
import sys
from main import setup_environment

def extract():
    # Helper to print list with 2 decimal precision
    def print_arr(name, arr):
        vals = [f"{x:.2f}" for x in arr]
        print(f"'{name}': [{', '.join(vals)}],")

    try:
        # Re-run setup to get the exact same profiles
        solar, load, prices, config = setup_environment(
            "electricityConsumptionAndProductioction.csv", 
            "PecanStreet_10_Homes_1Min_Data.csv"
        )
        
        print("PROFILES = {")
        print_arr('solar', solar)
        print_arr('load', load)
        print_arr('price', prices)
        print("}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract()
