from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject


PROMPT = """
Please autoformalize the following problem in Lean 4 with a header.
Use the following theorem names: my_favorite_theorem.\n\n
"""

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


def load_model(model_name: str):
    model = LLM(
        model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


def load_data(data_path: str):
    data = []
    with open(data_path, "rb") as f:
        for line in f.readlines():
            data.append(
                json.loads(line)
            )
    return data


def prepare_prompt(data: dict, tokenizer: AutoTokenizer) -> str:
    header = data["header"]
    problem = data["problem"]

    messages = [
        {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
        {"role": "user", "content": PROMPT + problem}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text


if __name__ == "__main__":
    # lean 4 repl
    config = LeanREPLConfig(project=TempRequireProject(lean_version="v4.19.0", require="mathlib"))
    server = LeanServer(config)
    response = server.run(Command(cmd=TEST_CMD))
    print("Test repl response:\n", response)
    
    # load data
    data_path = "datasets/formallite_combibench_proverbench.jsonl"
    data = load_data(data_path)

    # load model
    model_name = "AI-MO/Kimina-Autoformalizer-7B"
    model, tokenizer = load_model(model_name)

    # prepare prompts
    prompts = [prepare_prompt(sample, tokenizer) for sample in data]

    # autoformalize
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048
    )
    results = model.generate(prompts, sampling_params=sampling_params)
    formalizations = [r.outputs[0].text for r in results]

    print(prompts[0])
    print()
    print(formalizations[0])