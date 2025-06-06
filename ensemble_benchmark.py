from aggregator import RnnModel, states, load_rnn, predict_correct
import json

# ----- Benchmark Configuration ----

reward_model_name = "stepherd"  # Options: "stepherd", "rlhflow", "skywork", "armo"
dataset = "math500"
n = 50
reasoning_model_name = "llama31"
merging_strategy = "weighted_product"  # Options: "weighted_product", "weighted_sum"

# ------------------------------------


def ensemble_benchmark(scored_test_filename, model_path):
    with open(scored_test_filename, "r") as file:
        test_data = json.load(file)

    model = load_rnn(model_path)
    print("Model loaded from: ", model_path)
    print("Test data loaded from: ", scored_test_filename)
    print("Using merging strategy: ", merging_strategy)

    ratios = [i / 100 for i in range(0, 100, 1)]

    for ratio in ratios:
        correct = 0
        problem_count = 0
        for problem in test_data:
            verified_answers = problem["verified_answers"]
            # Timeouts are skipped
            if isinstance(verified_answers[0], str):
                continue
            problem_count += 1

            best_score = 0
            best_correct = False
            for answer in verified_answers:
                if isinstance(answer, str):
                    continue

                if ratio == 0:
                    answer["score"] = predict_correct(model, answer["steps"])
                if answer["score"] == 0:
                    score = 0
                else:
                    if merging_strategy == "weighted_product":
                        score = (answer["score"] ** ratio) * (
                            answer["score_prm"] ** (1 - ratio)
                        )
                    else:
                        score = answer["score"] * ratio + answer["score_prm"] * (
                            1 - ratio
                        )

                if score >= best_score:
                    best_score = score
                    best_correct = answer["correct"]

            if best_correct:
                correct += 1
        print(f"Ratio {ratio}:\t {correct}/{problem_count}")


if __name__ == "__main__":
    scored_test_filepath = f"results/trace_{dataset}_test_{n}_{reasoning_model_name}_scored_by_{reward_model_name}.json"
    model_path = (
        f"models/rnn_{dataset}_{n}_{reasoning_model_name}_best_normalized_accuracy.pth"
    )

    ensemble_benchmark(scored_test_filepath, model_path)
