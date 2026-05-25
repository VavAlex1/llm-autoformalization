from datasets import load_dataset
import os
import json
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject
from lean_interact.interface import LeanError
from tqdm import tqdm
import re
from eval import beql
import pandas as pd


TEST_CMD = """
import Mathlib
theorem algebra_539177 (a : Fin 2011 → ℝ) (ha : StrictMono a)                                                                                                                         
    (ha' : ∀ i, 0 < a i) :                                                                                                                                                
    ∃ i j, i < j ∧ a j - a i < ((1 + a i) * (1 + a j)) / 2010 := by                                                                                                                   
  sorry                                                                                                                                                                             
                                                                                                                                                                                      
theorem my_favorite_theorem (a : Fin 2011 → ℝ) (ha : StrictMono a)                                                                                                                    
    (ha' : ∀ i, 0 < a i) :                                                                                                                                                            
    ∃ i j, i < j ∧ a j - a i < ((1 + a i) * (1 + a j)) / 2010 := by                                                                                                                   
  exact?
"""


if __name__ == "__main__":
    # lean 4 repl
    config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.19.0", require="mathlib"))
    server = LeanServer(config)
    response = server.run(Command(cmd=TEST_CMD))
    print("Test repl response:\n", response)
    
    # load dataset 
    data_path = "benchmark_pred_processed.csv"
    df = pd.read_csv(data_path).iloc[:100]

    # prepare data
    beql_preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt = row["lean4_formalization"]
        pred = row["answers"]
        header = row["lean4_src_header"]

        beql_pred = beql(
            pred,
            gt,
            header,
            server
        )
        
        beql_preds.append(beql_pred)
    
    # save results
    df["beql"] = beql_preds
    df.to_csv("benchmark_beql.csv", index=False)
#    ds_test = ds_test.add_column("beql", beql_preds)
#    ds_test.to_csv("beql_result.csv", index=False)