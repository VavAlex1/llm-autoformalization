from vllm import LLM, SamplingParams
import os
from utils import find_boxed


check_prompt = """
Here is a natural language math problem and a translation in formal language Lean 4.
You need to carefully analyse these problems and figure out wether they are equivalent or not.
These problems must have exactly the same conditions and conclusions.
Mark false if they violate any requirement.
Also reply false if the formal statement is empty or malformed.

**Natural Language Problem**
{nlp}

```lean
{flp}
```

State your answer as $\\boxed{{true}}$ or $\\boxed{{false}}$ at the end of your response.
"""


def check_consistency(translations: list[dict], model: str, sampling_p: dict) -> list[float]:
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

    model = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=len(available_gpus)
    )

    sampling_params = SamplingParams(**sampling_p)

    prompts = [
       check_prompt.format(nlp=sample["nlp"], flp=sample["flp"]) for sample in translations
    ]

    outputs = model.generate(prompts, sampling_params)
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    answers = [output.outputs[0].text for output in outputs]

    consistency_results = [1.0 if find_boxed(ans) == 'true' else 0.0 for ans in answers]
    return consistency_results