# Given a solution to a problem, this script will divide the solution into individual steps using LLM.

from typing import List
from data_loader import math_n_loader
from models import MODEL, gpt4o_mini
import json
import re


def divide_step(solution: str) -> List[str]:
    # Divide the solution into individual steps
    with open("prompts/divide_steps.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("<solution>", solution)

    if_pass = False
    try_count = 0
    while not if_pass and try_count < 3:
        try:
            response = gpt4o_mini(prompt)
            # The response may be surrounded by ``` or ```json
            response = (
                re.sub(r"```(json|JSON)", "", response[0]).replace("```", "").strip()
            )
            steps = json.loads(response)
            if_pass = True
            break
        except:
            try_count += 1
            continue

    fallback_steps = [step for step in solution.split("\n") if step != ""]
    # make sure the return value has at least 1 element
    if fallback_steps == []:
        fallback_steps = [""]

    if not if_pass:
        # print("Warning: Failed to divide the solution into steps.")
        with open("failed_solution.txt", "a") as f:
            f.write(solution)
            f.write("\n" + "*" * 40 + "\n")
        return fallback_steps
    elif steps == []:
        return fallback_steps
    else:
        return steps


if __name__ == "__main__":
    math_data = math_n_loader()
    question = math_data[0]["question"] + "\nLet's think step by step."
    ans = MODEL(question)[0]
    steps = divide_step(ans)
    print(ans)
    print("-" * 40)
    print(json.dumps(steps, indent=4, ensure_ascii=False))
