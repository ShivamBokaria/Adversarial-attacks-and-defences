import os

def get_next_run_folder(base="results"):
    os.makedirs(base, exist_ok=True)

    runs = [int(d.split("_")[1]) for d in os.listdir(base) if d.startswith("run_")]
    next_id = max(runs) + 1 if runs else 1

    run_path = os.path.join(base, f"run_{next_id}")
    graphs = os.path.join(run_path, "graphs")
    tables = os.path.join(run_path, "tables")

    os.makedirs(graphs)
    os.makedirs(tables)

    return run_path, graphs, tables