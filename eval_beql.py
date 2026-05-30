from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject, AutoLeanServer
from lean_interact.interface import LeanError
from tqdm import tqdm
from eval import beql
import pandas as pd
import argparse


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
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-path", type=str, required=True)
    parser.add_argument("--output-data-path", type=str, required=True)
    parser.add_argument("--prediction-column", type=str, default="lean4_prediction")

    args = parser.parse_args()
    # lean 4 repl
    config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.19.0", require="mathlib"))
    server = AutoLeanServer(config)
    response = server.run(Command(cmd=TEST_CMD))
    print("Test repl response:\n", response)
    
    # load dataset 
    df = pd.read_csv(args.input_data_path).iloc[:100]
    print(f"Load {df.shape[0]} samples")

    # prepare data
    beql_preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        gt = row["lean4_formalization"]
        pred = row[args.prediction_column]
        header = row["lean4_src_header"]

        beql_pred = beql(
            pred,
            gt,
            header,
            server
        )
        
        beql_preds.append(beql_pred)

    # get final metric
    print(f"Average beql: {sum(beql_preds) / len(beql_preds)}")

    # save results
    df["beql"] = beql_preds
    df.to_csv(args.output_data_path, index=False)
    print(f"Save result to {args.output_data_path}")