from datasets import load_dataset
import os
import json
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject
from lean_interact.interface import LeanError
from tqdm import tqdm
import re
from eval import beql


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
    data_path = "PAug/ProofNetVerif"
    ds = load_dataset(data_path)
    ds_test = ds["test"]
    print("Example sample:\n", ds_test[0])

    # prepare data
    beql_preds = []
    for sample in tqdm(ds_test):
        gt = sample["lean4_formalization"]
        pred = sample["lean4_prediction"]
        header = sample["lean4_src_header"]

        beql_pred = beql(
            pred,
            gt,
            header,
            server
        )
        
        beql_preds.append(beql_pred)
    
    # save results
    ds_test = ds_test.add_column("beql", beql_preds)
    ds_test.to_csv("beql_result.csv", index=False)