import pandas as pd

def save_results_table(results, path):
    df = pd.DataFrame(results).T
    df.to_csv(f"{path}/results.csv")