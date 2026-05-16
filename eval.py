from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
from lean_interact import LeanREPLConfig, LeanServer, Command, TempRequireProject
from lean_interact.interface import LeanError
from tqdm import tqdm
import re


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


def prepare_prompt(data: dict, tokenizer: AutoTokenizer):
    header = data["header"].strip()
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

    prefix = header + "\n" + "theorem my_favorite_theorem "
    prompt = text + "\n" + prefix

    return prompt, prefix


def get_formalization(output, prefix):
    return prefix + " " + output


def check_syntax(formalization: str, server: LeanServer):
    try:
        formalization = formalization.strip()
        response = server.run(Command(cmd=formalization))
        if isinstance(response, LeanError):
            return False
        else:
            for message in response.messages:
                if message.severity == "error":
                    return False
        return True
    except Exception:
        return False


def check_identic(
    formalization_1,
    formalization_2,
    header,
    server
):
    formalization_1 = prepare_formalization(
        formalization_1,
        header,
        add_sorry=True,
        add_exact=False
    )
    formalization_2 = prepare_formalization(
        formalization_2,
        header,
        add_sorry=False,
        add_exact=True
    )

    code = header + "\n" + formalization_1 + "\n\n" + formalization_2 + "\n"
    response = server.run(Command(cmd=code))

    name = extract_theorem_name(formalization_1).strip()
    for message in response.messages:
        if message.severity == 'error':
            return False
        if (
            message.severity == "info"
            and message.data.startswith("Try this:")
            and name in message.data
        ):
            return True
    
    return False


THEOREM_NAME_PATTERN = re.compile('theorem\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*')
def extract_theorem_name(lean_code: str) -> str:
    match = THEOREM_NAME_PATTERN.search(lean_code)
    if match:
        return match.group(1)
    assert False, f"No theorem name in:\n {lean_code}"


def prepare_formalization(
    formalization: str,
    header,
    add_sorry: bool,
    add_exact: bool
    ) -> str:
    formalization = formalization.removeprefix(header)
    formalization = formalization.rsplit(":=", 1)[0] + ":= by\n"

    if add_sorry:
        formalization += "sorry"
    elif add_exact:
        formalization += "exact?"
    
    return formalization


def beql(
    formalization: str,
    ground_truth: str,
    header: str,
    server: LeanServer
):  
    check_1 = check_identic(formalization, ground_truth, header, server)
    check_2 = check_identic(ground_truth, formalization, header, server)
    return check_1 and check_2


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
    prompts, prefixes = [], []
    for sample in data:
        prompt, prefix = prepare_prompt(sample, tokenizer)
        prompts.append(prompt)
        prefixes.append(prefix)
    
    # autoformalize
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048
    )
    results = model.generate(prompts, sampling_params=sampling_params)

    # get final formalizations
    outputs = [r.outputs[0].text for r in results]
    formalizations = [
        get_formalization(output, prefix) for output, prefix in zip(outputs, prefixes)
    ]

    # count syntax pass rate
    count = 0
    for formalization in tqdm(formalizations):
        count += check_syntax(formalization, server)
    print("syntax pass rate: ", count / len(formalizations))

    # count beql metric
    count = 0
    for formalization, sample in tqdm(zip(formalizations, data), total=len(data)):
        gt = sample["verified_code"]
        header = sample["header"]
        count += beql(
            formalization,
            gt,
            header,
            server
        )
    print("beql pass rate: ", count / len(formalizations))