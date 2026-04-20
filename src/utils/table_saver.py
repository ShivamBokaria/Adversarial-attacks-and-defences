import pandas as pd

def save_results_table(results, save_path):
    df = pd.DataFrame(results).T
    df.to_csv(f"{save_path}/results.csv")