
import pandas as pd
import matplotlib.pyplot as plt
from evaluation import create_summary_table

def report():
    print("RECOVERING SIMULATION RESULTS...")
    
    # 1. Load Evaluation Results
    try:
        from evaluation import run_comprehensive_evaluation
        # We can't easily re-load the dict structure from CSV without parsing.
        # But we can read the CSV directly.
        df = pd.read_csv("results/evaluation_results.csv")
        print("\n=== EVALUATION SUMMARY ===")
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not load evaluation results: {e}")

    # 2. Re-calculate Robustness Summary (if not saved)
    # The robustness results dataframe isn't saved by default in main.py, only plotting.
    # But we can try to find if we can reconstruct or if I need to re-run robustness test.
    # Wait, the main.py logic is:
    # robustness_results = run_robustness_test(...)
    # robustness_fig = plot_robustness_comparison(...)
    
    # Check if we can just re-run the robustness SUMMARY part if we had saved the data? 
    # The data wasn't saved to CSV in main.py (only plot).
    # However, the user saw the 'Robustness Summary' in the output before the crash!
    # "ROBUSTNESS SUMMARY ... Agent ... R-SAC ..."
    # So we know R-SAC did well.
    
    # Let's just print the evaluation results CSV which is robustly saved.
    pass

if __name__ == "__main__":
    report()
