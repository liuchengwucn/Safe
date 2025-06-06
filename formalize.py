from typing import List
from models import gpt4o
from data_loader import math_loader


def formalize(question: str, current_state: List[str], step: str) -> str | None:
    with open("prompts/formalize.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("<question>", question)
    prompt = prompt.replace("<answer>", "\n".join(current_state))
    prompt = prompt.replace("<step>", step)

    response = gpt4o(prompt)
    if "false" in response[0].lower():
        return None

    return response[0]


if __name__ == "__main__":
    data = math_loader()["train"]
    for sample in data:
        if (
            "algebra" not in sample["type"].lower()
            and "number" not in sample["type"].lower()
        ):
            continue
        question, answer = sample["question"], sample["answer"]
        print(f"### Question: {question}")
        answer = answer.split(".")
        for i, step in enumerate(answer):
            step = step.strip()
            if not step:
                continue
            print(f"### Step:\n{step}")
            print(f"### Formalized:\n{formalize(question, answer[:i], step)}")
            print()
        print("\n\n")
        input()
