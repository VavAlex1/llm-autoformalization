"""
Прогон локальной модели автоформализации (AI-MO/Kimina-Autoformalizer-7B)
через vLLM на датасете AlexVav01/FormalMath-formalization.

Хэдер из датасета (lean4_src_header) вставляется в промпт как префикс ответа
ассистента, после чего модель продолжает генерацию уже с него. В поле
"lean4_prediction" сохраняется только теорема (без хэдера).
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."
USER_PROMPT = (
    "Please autoformalize the following problem in Lean 4 with a header. "
    "Use the following theorem names: my_favorite_theorem.\n\n"
)
THEOREM_PREFIX = "theorem my_favorite_theorem "


def build_prompt(problem: str, header: str, tokenizer):
    """Промпт для vLLM (с хэдером, чтобы модель его использовала)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT + problem.strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = text + header.strip() + "\n\n" + THEOREM_PREFIX
    return prompt


if __name__ == "__main__":
    dataset_name = "AlexVav01/FormalMath-formalization"
    model_name = "AI-MO/Kimina-Autoformalizer-7B"
    output_path = "kimina_formalizations.csv"

    # данные
    dataset = load_dataset(dataset_name, split="train")

    # модель + токенайзер
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = LLM(model_name)

    # промпты с вставленным хэдером
    prompts = [
        build_prompt(sample["nl_statement"], sample["lean4_src_header"], tokenizer)
        for sample in dataset
    ]

    # генерация
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=2048)
    results = model.generate(prompts, sampling_params=sampling_params)

    # финальная формализация = theorem my_favorite_theorem + генерация (без хэдера)
    predictions = [
        THEOREM_PREFIX + result.outputs[0].text for result in results
    ]

    # сохранение в поле lean4_prediction
    dataset = dataset.add_column("lean4_prediction", predictions)
    dataset.to_csv(output_path)
    print(f"Сохранено {len(dataset)} записей с полем 'lean4_prediction' в {output_path}")
