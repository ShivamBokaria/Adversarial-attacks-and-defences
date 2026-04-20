import os

def get_next_run_folder(base_path="results"):
    os.makedirs(base_path, exist_ok=True)

    existing = [d for d in os.listdir(base_path) if d.startswith("run_")]
    run_nums = [int(d.split("_")[1]) for d in existing if "_" in d]

    next_run = max(run_nums, default=0) + 1

    run_path = os.path.join(base_path, f"run_{next_run}")
    graphs_path = os.path.join(run_path, "graphs")
    tables_path = os.path.join(run_path, "tables")

    os.makedirs(graphs_path)
    os.makedirs(tables_path)

    return run_path, graphs_path, tables_path