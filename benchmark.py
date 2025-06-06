from aggregator import RnnModel, states, load_rnn, predict_correct
from tqdm import tqdm
import json

# ----- Benchmark Configuration ----

use_prm = True  # Set to True to merge a reward model score and our aggregator score
reward_model_name = "stepherd"  # Options: "stepherd", "rlhflow", "skywork", "armo"

dataset = "math500"
sample_count = 5
n = 50
reasoning_model_name = "llama31"

# ------------------------------------


def majority_and_pass(test_data, problem_count):
    """
    Print the number of problems with at least one correct answer (Pass@k)
    and the number of problems with a majority of correct answers (Majority@k).
    """
    correct_count = 0
    majority_correct_count = 0
    for problem in test_data:
        verified_answers = problem["verified_answers"]
        # Timeouts are skipped
        if isinstance(verified_answers[0], str):
            continue

        if any(answer["correct"] for answer in verified_answers):
            correct_count += 1
        if (
            sum(answer["correct"] for answer in verified_answers)
            >= len(verified_answers) / 2
        ):
            majority_correct_count += 1
    print(f"Any correct count: {correct_count}/{problem_count}")
    print(f"Majority correct count: {majority_correct_count}/{problem_count}")


def benchmark(model_path, test_data):
    if use_prm:
        if reward_model_name == "stepherd":
            from rm_stepherd import reward_model
        elif reward_model_name == "rlhflow":
            from rm_rlhflow import reward_model
        elif reward_model_name == "skywork":
            from rm_skywork import reward_model
        elif reward_model_name == "armo":
            from rm_armo import reward_model

    model = load_rnn(model_path)

    print("Model loaded from: ", model_path)

    # Use the first i verified answers for each problem
    # Setting i to values smaller than sample_count is useful
    # when you want to test multiple n with one trace (with a large sample_count)
    # For example, if sample_count=32, you can set i to one of [1, 2, 4, 8, 16, 32]

    for i in [sample_count]:
        baseline_correct_count = 0
        model_correct_count = 0
        problem_count = 0

        prm_correct_count = 0
        plus_correct_count = 0
        mul_correct_count = 0
        min_correct_count = 0
        max_correct_count = 0

        for problem in tqdm(test_data):
            verified_answers = problem["verified_answers"][:i]
            # Timeouts are skipped
            if isinstance(verified_answers[0], str):
                continue

            # baseline_correct is the average correct rate of all verified answers
            baseline_correct_count += sum(
                [answer["correct"] for answer in problem["verified_answers"]]
            ) / len(problem["verified_answers"])

            problem_count += 1

            best_score = 0
            best_score_prm = 0
            best_score_plus = 0
            best_score_mul = 0
            best_score_min = 0
            best_score_max = 0

            best_correct = False
            best_correct_prm = False
            best_correct_plus = False
            best_correct_mul = False
            best_correct_min = False
            best_correct_max = False

            for answer in verified_answers:
                # Our retrospective score
                score = predict_correct(model, answer["steps"])
                answer["score"] = score

                if score >= best_score:
                    best_score = score
                    best_correct = answer["correct"]

                if use_prm:
                    # Prospective score from a reward model
                    score_prm = reward_model(
                        problem["question"], [step["step"] for step in answer["steps"]]
                    )
                    answer["score_prm"] = score_prm

                    # Several ways to merge the scores
                    # For finer control, use ensemble_benchmark.py instead
                    score_plus = score + score_prm
                    score_mul = score * score_prm
                    score_min = min(score, score_prm)
                    score_max = max(score, score_prm)

                    if score_prm >= best_score_prm:
                        best_score_prm = score_prm
                        best_correct_prm = answer["correct"]

                    if score_plus >= best_score_plus:
                        best_score_plus = score_plus
                        best_correct_plus = answer["correct"]

                    if score_mul >= best_score_mul:
                        best_score_mul = score_mul
                        best_correct_mul = answer["correct"]

                    if score_min >= best_score_min:
                        best_score_min = score_min
                        best_correct_min = answer["correct"]

                    if score_max >= best_score_max:
                        best_score_max = score_max
                        best_correct_max = answer["correct"]

            model_correct_count += best_correct
            prm_correct_count += best_correct_prm
            plus_correct_count += best_correct_plus
            mul_correct_count += best_correct_mul
            min_correct_count += best_correct_min
            max_correct_count += best_correct_max

        print(f"Top {i} reward")
        print(f"Baseline correct count: {baseline_correct_count:.0f}/{problem_count}")
        print(f"Model correct count: {model_correct_count}/{problem_count}")
        majority_and_pass(test_data, problem_count)
        if use_prm:
            print(f"PRM correct count: {prm_correct_count}/{problem_count}")
            print(f"Plus correct count: {plus_correct_count}/{problem_count}")
            print(f"Mul correct count: {mul_correct_count}/{problem_count}")
            print(f"Min correct count: {min_correct_count}/{problem_count}")
            print(f"Max correct count: {max_correct_count}/{problem_count}")

            scored_test_filename = (
                f"{test_filepath[:-5]}_scored_by_{reward_model_name}.json"
            )
            with open(scored_test_filename, "w") as file:
                json.dump(test_data, file)


if __name__ == "__main__":
    # Train your own aggregator model with aggregator.py
    model_path = (
        f"models/rnn_{dataset}_{n}_{reasoning_model_name}_best_normalized_accuracy.pth"
    )
    test_filepath = f"results/trace_{dataset}_test_{n}_{reasoning_model_name}.json"

    with open(test_filepath, "r") as file:
        test_data = json.load(file)

    benchmark(model_path, test_data)
