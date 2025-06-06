import json
import os
from typing import List, Dict

MATH_PATH = "datasets/MATH"
GSM8K_PATH = "datasets/gsm8k"
MATH500_PATH = "datasets/prm800k"


def extract_boxed_answer(text: str) -> str:
    start = text.find("\\boxed{")

    if start == -1:
        start = text.find("\\boxed ")
        start += 7
        end = start + 1
        while text[end].isdigit():
            end += 1
        return text[start:end]

    start += 7
    bracket_count = 1
    end = start

    while bracket_count > 0 and end < len(text):
        if text[end] == "{":
            bracket_count += 1
        elif text[end] == "}":
            bracket_count -= 1

        end += 1
        if bracket_count == 0:
            return text[start : end - 1]


def extract_number_sign_answer(text: str) -> str:
    start = text.find("#### ")
    start += 5
    return text[start:]


def math_loader() -> Dict[str, List[Dict[str, str]]]:
    # part: train, test
    # dict keys: 'problem', 'question', 'level', 'type', 'solution', 'ground_truth', 'unique_id'
    math_data = {}
    for part in os.listdir(MATH_PATH):
        if not os.path.isdir(os.path.join(MATH_PATH, part)):
            continue

        part_data = []
        for type in os.listdir(os.path.join(MATH_PATH, part)):
            if not os.path.isdir(os.path.join(MATH_PATH, part, type)):
                continue

            for file in os.listdir(os.path.join(MATH_PATH, part, type)):
                if not file.endswith(".json"):
                    continue

                with open(os.path.join(MATH_PATH, part, type, file), "r") as f:
                    data = json.load(f)
                    ground_truth = extract_boxed_answer(data["solution"])
                    data["ground_truth"] = ground_truth
                    data["question"] = data["problem"]  # alias
                    data["answer"] = data["solution"]  # alias
                    data["unique_id"] = f"math_{part}_{type}_{file.split('.')[0]}"
                    part_data.append(data)

        math_data[part] = part_data

    return math_data


def math_500_loader() -> Dict[str, List[Dict[str, str]]]:
    math_data = {}

    filepath = os.path.join(MATH500_PATH, "math_splits")

    for part in os.listdir(filepath):
        part_name = part.split(".")[0]
        part_data = []
        with open(os.path.join(filepath, part), "r") as f:
            for line in f:
                data = json.loads(line)
                data["ground_truth"] = data["answer"]
                data["question"] = data["problem"]  # alias
                data["answer"] = data["solution"]  # alias
                part_data.append(data)

        math_data[part_name] = part_data

    return math_data


def math_n_loader(n=50) -> List[Dict[str, str]]:
    math_data = math_500_loader()
    math_data = math_data["test"][:n]

    return math_data


def math_train_loader(n=50) -> List[Dict[str, str]]:
    math_data = math_loader()
    math_data = math_data["train"][:n]

    return math_data


def gsm8k_train_loader(n=None) -> List[Dict[str, str]]:
    gsm8k_data = gsm8k_loader()
    if n is not None:
        gsm8k_data = gsm8k_data["train"][:n]
    else:
        gsm8k_data = gsm8k_data["train"]

    return gsm8k_data


def gsm8k_test_loader(n=None) -> List[Dict[str, str]]:
    gsm8k_data = gsm8k_loader()
    if n is not None:
        gsm8k_data = gsm8k_data["test"][:n]
    else:
        gsm8k_data = gsm8k_data["test"]

    return gsm8k_data


def gsm8k_loader() -> Dict[str, List[Dict[str, str]]]:
    # part: train, test, train_socratic, test_socratic
    # dict keys: 'question', 'answer', 'ground_truth', 'unique_id'
    gsm8k_data = {}
    for part in os.listdir(GSM8K_PATH):
        if not part.endswith(".jsonl") or "model" in part:
            continue

        part_data = []
        part_name = part.split(".")[0]
        with open(os.path.join(GSM8K_PATH, part), "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                ground_truth = extract_number_sign_answer(data["answer"])
                data["ground_truth"] = ground_truth
                data["unique_id"] = f"gsm8k_{part_name}_{i}"
                part_data.append(data)

        gsm8k_data[part_name] = part_data

    return gsm8k_data


def collegemath_train_loader(n=None) -> List[Dict[str, str]]:
    collegemath_data = collegemath_loader()
    if n is not None:
        collegemath_data = collegemath_data["train"][:n]
    else:
        collegemath_data = collegemath_data["train"]

    return collegemath_data


def collegemath_test_loader(n=None) -> List[Dict[str, str]]:
    collegemath_data = collegemath_loader()
    if n is not None:
        collegemath_data = collegemath_data["test"][:n]
    else:
        collegemath_data = collegemath_data["test"]

    return collegemath_data


def collegemath_loader() -> Dict[str, List[Dict[str, str]]]:
    test_filepath = "datasets/collegemath/full_test.jsonl"
    train_filepath = "datasets/collegemath/full_train.jsonl"
    test_data = []
    train_data = []

    with open(test_filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["data_source"].startswith("college_math"):
                new_data = {}
                new_data["question"] = data["question"]
                new_data["ground_truth"] = data["answer"]
                new_data["unique_id"] = data["question_number"]
                test_data.append(new_data)

    with open(train_filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["data_source"].startswith("college_math"):
                new_data = {}
                new_data["question"] = data["question"]
                new_data["ground_truth"] = data["answer"]
                new_data["unique_id"] = data["question_number"]
                train_data.append(new_data)

    return {"train": train_data, "test": test_data}


if __name__ == "__main__":
    gsm8k_data = gsm8k_loader()
    print(
        "GSM8K data loaded successfully. Length of train set:", len(gsm8k_data["train"])
    )
    math_data = math_loader()
    print(
        "Math data loaded successfully. Length of train set:", len(math_data["train"])
    )
    math500_data = math_500_loader()
    print(
        "Math500 data loaded successfully. Length of train set:",
        len(math500_data["train"]),
    )
    math_train_data = math_train_loader(n=50)
    print("Math train data loaded successfully. Length:", len(math_train_data))
    collegemath_data = collegemath_loader()
    print(
        "CollegeMath data loaded successfully. Length of train set:",
        len(collegemath_data["train"]),
    )
