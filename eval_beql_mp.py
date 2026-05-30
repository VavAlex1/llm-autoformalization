from lean_interact import LeanREPLConfig, TempRequireProject, AutoLeanServer
from tqdm import tqdm
from eval import beql
import pandas as pd
import argparse
import multiprocessing as mp


SERVER = None


def init_worker(config):
    global SERVER
    SERVER = AutoLeanServer(config)


def worker(task):
    row, prediction_column = task

    gt = row["lean4_formalization"]
    pred = row[prediction_column]
    header = row["lean4_src_header"]

    return beql(
        pred,
        gt,
        header,
        SERVER,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-data-path", type=str, required=True)
    parser.add_argument("--output-data-path", type=str, required=True)
    parser.add_argument("--prediction-column", type=str, default="lean4_prediction")
    parser.add_argument("--num-processes", type=int, default=4)

    args = parser.parse_args()
    # lean 4 repl
    config = LeanREPLConfig(
        project=TempRequireProject(
            lean_version="v4.19.0",
            require="mathlib",
        )
    )
    print(f"Using {args.num_processes} processes")
    
    # load dataset 
    df = pd.read_csv(args.input_data_path)
    print(f"Load {df.shape[0]} samples")

    # prepare data
    tasks = [
        (row, args.prediction_column)
        for row in df.to_dict("records")
    ]
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.num_processes,
        initializer=init_worker,
        initargs=(config,),
    ) as pool:
        beql_preds = list(
            tqdm(
                pool.imap(worker, tasks),
                total=len(tasks),
            )
        )

    # get final metric
    print(f"Average beql: {sum(beql_preds) / len(beql_preds)}")

    # save results
    df["beql"] = beql_preds
    df.to_csv(args.output_data_path, index=False)
    print(f"Save result to {args.output_data_path}")