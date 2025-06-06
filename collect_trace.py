from typing import List, Tuple
from data_loader import (
    math_n_loader,
    math_train_loader,
    gsm8k_test_loader,
    gsm8k_train_loader,
    collegemath_train_loader,
    collegemath_test_loader,
)
from divide_steps import divide_step
from models import llama, gpt4o, gpt4o_mini, llama30, deepseek_math
from verifier import Verifier
import json
from tqdm import tqdm
import concurrent.futures
from utils import timeout_handler
import os

# ----- Configuration ----

# Choose one from the following models:
# Options: gpt4o, llama30, llama31, deepseek_math
reasoning_model_name = "llama31"

# Number of samples to generate for each question (Best of N)
sample_count = 5

# Number of data samples
n = 50

# Dataset to use
# Options: math500, gsm8k, collegemath
dataset = "math500"

# Data split to use
# Options: train, test
data_split = "train"

prover = "deepseek_prover"  # Options: deepseek_prover, copra

verbose = False

# -------------------------


def verbose_print(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def check_answer(question: str, answer: List[str], ground_truth: str) -> bool:
    with open("prompts/check_answer.txt", "r") as f:
        prompt = f.read()

    prompt = prompt.replace("<question>", question)
    prompt = prompt.replace("<answer>", "\n".join(answer))
    prompt = prompt.replace("<ground_truth>", ground_truth)

    response = gpt4o_mini(prompt)
    if "true" in response[0].lower():
        return True
    return False


def predict_divide_steps(question: str, ground_truth: str) -> Tuple[List[str], bool]:
    v = Verifier()
    answers = reasoning_model(
        question + "\n Let's think step by step.", temperature=0.5, n=sample_count
    )

    verified_answers = []

    def verify_answer(answer):
        verbose_print("Verifying answer:", answer)
        steps = divide_step(answer)
        correct = check_answer(question, steps, ground_truth)
        verified_steps = []

        verbose_print("Divided steps:", len(steps))
        for i in range(len(steps)):
            verbose_print("Verifying step:", i)
            # Four possible states for each step:
            # Success of Proof
            # Failure of Proof
            # No Verification Required
            # Failed Formalization

            formal_statement = None
            verified = False
            state = None
            proof = None

            # Try to formalize the step 3 times maximum
            formalize_try_count = 0
            while formalize_try_count < 3:
                formal_statement = v.formalize(question, steps[:i], steps[i])
                if formal_statement is None:
                    break
                verified = v.lean_verifier(formal_statement)
                if verified:
                    break
                formalize_try_count += 1

            # If the formal statement is None, it means no verification is required
            if formal_statement is None:
                state = "No Verification Required"
            # If the formal statement is not None and not verified, it means the formalization failed
            elif not verified:
                state = "Failed Formalization"
            # If the formal statement is verified, we attempt to prove it
            else:
                if_pass, proof = prove_and_check(formal_statement)
                if if_pass:
                    state = "Success of Proof"
                else:
                    state = "Failure of Proof"

            verified_steps.append(
                {
                    "step": steps[i],
                    "formal_statement": formal_statement,
                    "proof": proof,
                    "state": state,
                    "correct": correct,
                }
            )

        return {
            "answer": answer,
            "steps": verified_steps,
            "correct": correct,
        }

    for answer in answers:
        verified_answer = timeout_handler(
            verify_answer,
            args=(answer,),
            timeout_duration=3600,
            default={
                "answer": answer,
                "steps": [],
                "correct": check_answer(question, divide_step(answer), ground_truth),
            },
        )
        verified_answers.append(verified_answer)
    return verified_answers


def predict_helper(data_sample):
    return timeout_handler(
        predict_divide_steps,
        args=(data_sample["question"], data_sample["ground_truth"]),
        timeout_duration=3600 * sample_count,
        default=["predict timeout"],
    )


if __name__ == "__main__":
    if reasoning_model_name == "gpt4o":
        reasoning_model = gpt4o
    elif reasoning_model_name == "llama30":
        reasoning_model = llama30
    elif reasoning_model_name == "llama31":
        reasoning_model = llama
    elif reasoning_model_name == "deepseek_math":
        reasoning_model = deepseek_math

    if dataset == "math500" and data_split == "test":
        math_data = math_n_loader(n=n)
    elif dataset == "math500" and data_split == "train":
        math_data = math_train_loader(n=n)
    elif dataset == "gsm8k" and data_split == "test":
        math_data = gsm8k_test_loader(n=n)
    elif dataset == "gsm8k" and data_split == "train":
        math_data = gsm8k_train_loader(n=n)
    elif dataset == "collegemath" and data_split == "test":
        math_data = collegemath_test_loader(n=n)
    elif dataset == "collegemath" and data_split == "train":
        math_data = collegemath_train_loader(n=n)
    else:
        raise ValueError("Invalid dataset or data split specified.")

    if prover == "deepseek_prover":
        from deepseek_prover_helper import prove_and_check
    elif prover == "copra":
        from copra_helper import prove_and_check

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(n if n is not None else 16, 16)
    ) as executor:
        predict_results = list(
            tqdm(
                executor.map(predict_helper, math_data),
                total=len(math_data),
            )
        )

    for i, problem in enumerate(math_data):
        verified_answers = predict_results[i]
        problem["verified_answers"] = verified_answers

    if not os.path.exists("results"):
        os.makedirs("results")
    with open(
        f"results/trace_{dataset}_{data_split}_{n}_{reasoning_model_name}.json", "w"
    ) as f:
        json.dump(math_data, f, indent=4, ensure_ascii=False)
