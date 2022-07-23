import argparse
import json
import os
from collections import defaultdict
import pandas as pd

#def save_indices_for_overlap(args: argparse.Namespace, total: int, steps: int, selected_top_k_runs: list) -> None:
def save_indices_for_overlap(args, selected_top_k_runs: list) -> None:
    selected_top_k_dict = defaultdict(list)
    filePrefix = os.getenv("FILE_PREFIX")
    for i in range(len(selected_top_k_runs)):
        top_k_run = selected_top_k_runs[i]
        selected_top_k_dict[i] = top_k_run

    with open(f"{os.getenv('INDICES_PATH')}/top_k_{args.task}_{args.acquisition}-{filePrefix}.json", "w") as f:
        json.dump(selected_top_k_dict, f)
    pool_path = os.getenv("TREC_POOL")
    X = list(pd.read_csv(pool_path)["Description"].values)
    for i in range(len(selected_top_k_runs)):
        top_k_run = selected_top_k_runs[i]
        with open("top_k_cartography_{}".format(filePrefix)) as f:
            for j in top_k_run:
                f.write(X[j])
        
