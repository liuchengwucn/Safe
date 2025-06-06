from typing import List
from torch import nn
import torch
import json

# ----- Aggregator Configuration ----

states = [
    "Failure of Proof",
    "Success of Proof",
    "No Verification Required",
    "Failed Formalization",
]
input_size = len(states)
output_size = 1
hidden_size = 64

# ------ Training Configuration ------

batch_size = 32
threshold = 0.5
learning_rate = 0.0001
num_epochs = 200

# ------- Dataset Configuration -------

# Refer to collect_trace.py for dataset trace generation
reasoning_model_name = "llama31"
dataset = "math500"
sample_count = 50

# -------------------------------------


class RnnModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RnnModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x'size looks like this: [seq_len, input_size]
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output[:, -1, :]


def prepare_data_pair(data_set):
    X_pos = []
    X_neg = []

    for problem_pos, problem_neg in data_set:
        steps_pos = problem_pos["steps"]
        steps_neg = problem_neg["steps"]

        if len(steps_pos) == 0 or len(steps_neg) == 0:
            continue
        # use one-hot encoding for the input
        X_pos.append(
            [
                [1.0 if state == step["state"] else 0.0 for state in states]
                for step in steps_pos
            ]
        )
        X_neg.append(
            [
                [1.0 if state == step["state"] else 0.0 for state in states]
                for step in steps_neg
            ]
        )

    return prepare_tensor_pair(X_pos, X_neg)


def prepare_tensor_pair(X_pos, X_neg):
    X_pos = [torch.tensor(x).unsqueeze(0).cuda() for x in X_pos]
    X_neg = [torch.tensor(x).unsqueeze(0).cuda() for x in X_neg]
    return X_pos, X_neg


def train_rnn(problems: List):
    # Each element in problems is as follows:
    # state filed could only be "Failure of Proof", "Success of Proof", "No Verification Required", "Failed Formalization"
    # {
    #     "answer": "The point $(0,3)$ in rectangular coordinates can be converted to polar coordinates by finding the radius $r$ and the angle $\theta$.",
    #     "correct": True,
    #     steps: [
    #         {
    #             "state": "Failure of Proof",
    #         },
    #         {
    #             "state": "Success of Proof",
    #         },
    #         {
    #             "state": "No Verification Required",
    #         }
    #     ]
    # }

    # 80% of the data will be used for training and 20% for testing (validation)
    train_set = problems[: int(0.8 * len(problems))]
    test_set = problems[int(0.8 * len(problems)) :]

    test_pairs = []
    for problem in train_data:
        correct_answers = []
        incorrect_answers = []

        for answer in problem["verified_answers"]:
            if not isinstance(answer, str) and len(answer["steps"]) > 0:
                if answer["correct"]:
                    correct_answers.append(answer)
                else:
                    incorrect_answers.append(answer)

        for correct_answer in correct_answers:
            for incorrect_answer in incorrect_answers:
                test_pairs.append((correct_answer, incorrect_answer))
    test_pairs = test_pairs[int(0.8 * len(test_pairs)) :]

    X_pos, X_neg = prepare_data_pair(test_pairs)

    def prepare_data(data_set):
        X = []
        y = []
        for problem in data_set:
            steps = problem["steps"]
            if len(steps) == 0:
                continue
            # use one-hot encoding for the input
            X.append(
                [
                    [1.0 if state == step["state"] else 0.0 for state in states]
                    for step in steps
                ]
            )
            y.append(1 if problem["correct"] else 0)
        return prepare_tensor(X, y)

    def prepare_tensor(X, y):
        X = [torch.tensor(x).unsqueeze(0).cuda() for x in X]
        y = [torch.tensor([yi]).float().unsqueeze(0).cuda() for yi in y]
        return X, y

    def positive_negative_weight(data_set):
        count = 0
        for problem in data_set:
            if problem["correct"]:
                count += 1
        positive_ratio = count / len(data_set)
        negative_ratio = 1 - positive_ratio

        positive_weight = 0.5 / positive_ratio
        negative_weight = 0.5 / negative_ratio

        slope = positive_weight - negative_weight
        bias = negative_weight

        return slope, bias

    X_train, y_train = prepare_data(train_set)
    X_test, y_test = prepare_data(test_set)
    slope, bias = positive_negative_weight(train_set)

    # Train the model with gpu
    model = RnnModel(input_size, hidden_size, output_size).cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0
    best_normalized_accuracy = 0
    best_pair_accuracy = 0
    best_f1 = 0
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0

        train_indices = torch.randperm(len(X_train))
        # train_indices = range(len(X_train))
        for i in train_indices:
            x = X_train[i]
            y = y_train[i]
            output = model(x)
            loss = criterion(output, y)

            weight = y * slope + bias
            loss = loss * weight

            avg_loss += loss.item()
            loss.backward()
            if i % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            pair_correct = 0

            y_1_output_1 = 0
            y_1_output_0 = 0
            y_0_output_1 = 0
            y_0_output_0 = 0

            train_correct = 0
            for i in range(len(X_train)):
                x = X_train[i]
                y = y_train[i]
                output = model(x)
                x = nn.Sigmoid()(output)
                if (output > threshold and y == 1) or (output <= threshold and y == 0):
                    train_correct += 1

            for i in range(len(X_test)):
                x = X_test[i]
                y = y_test[i]
                output = model(x)
                x = nn.Sigmoid()(output)
                total += 1
                if (output > threshold and y == 1) or (output <= threshold and y == 0):
                    correct += 1

                if y == 1 and output > threshold:
                    y_1_output_1 += 1
                elif y == 1 and output <= threshold:
                    y_1_output_0 += 1
                elif y == 0 and output > threshold:
                    y_0_output_1 += 1
                else:
                    y_0_output_0 += 1

            for i in range(len(test_pairs)):
                x_pos = X_pos[i]
                x_neg = X_neg[i]
                output_pos = model(x_pos)
                output_neg = model(x_neg)
                if output_pos > output_neg:
                    pair_correct += 1

            train_accuracy = train_correct / len(X_train)
            accuracy = correct / total
            pair_accuracy = pair_correct / len(test_pairs)

            normalized_accuracy = (
                y_1_output_1 / (y_1_output_1 + y_1_output_0)
                + y_0_output_0 / (y_0_output_0 + y_0_output_1)
            ) / 2

            precision = y_1_output_1 / max((y_1_output_1 + y_0_output_1), 1)
            recall = y_1_output_1 / max(1, (y_1_output_1 + y_1_output_0), 1)
            if precision + recall == 0:
                f1 = 0
            else:
                f1 = 2 * precision * recall / (precision + recall)

            print(
                f"Epoch {epoch + 1}, Train Accuracy: {train_correct}/{len(X_train)}={train_accuracy:.2f}, Test Accuracy: {correct}/{total}={accuracy:.2f}, Pair Accuracy: {pair_correct}/{len(test_pairs)}={pair_accuracy:.2f}, Loss: {avg_loss:.2f}, Normalized Accuracy: {normalized_accuracy:.2f}, F1: {f1:.2f}"
            )
            print(f"y=1, output=1: {y_1_output_1}")
            print(f"y=1, output=0: {y_1_output_0}")
            print(f"y=0, output=1: {y_0_output_1}")
            print(f"y=0, output=0: {y_0_output_0}")

            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                torch.save(
                    model.state_dict(),
                    f"models/rnn_{dataset}_{sample_count}_{reasoning_model_name}_best_accuracy.pth",
                )

            # We use normalized accuracy by default
            if normalized_accuracy >= best_normalized_accuracy:
                best_normalized_accuracy = normalized_accuracy
                torch.save(
                    model.state_dict(),
                    f"models/rnn_{dataset}_{sample_count}_{reasoning_model_name}_best_normalized_accuracy.pth",
                )

            if pair_accuracy >= best_pair_accuracy:
                best_pair_accuracy = pair_accuracy
                torch.save(
                    model.state_dict(),
                    f"models/rnn_{dataset}_{sample_count}_{reasoning_model_name}_best_pair_accuracy.pth",
                )

            if f1 >= best_f1:
                best_f1 = f1
                torch.save(
                    model.state_dict(),
                    f"models/rnn_{dataset}_{sample_count}_{reasoning_model_name}_best_f1.pth",
                )

            torch.save(
                model.state_dict(),
                f"models/rnn_{dataset}_{sample_count}_{reasoning_model_name}_last.pth",
            )

            if train_accuracy - accuracy > 0.2 and train_accuracy > 0.9:
                early_stop_count += 1
            else:
                early_stop_count = 0
            if early_stop_count >= 5:
                break

    print("-" * 80)
    print("Training finished")
    print(f"batch_size = {batch_size}")
    print(f"learning_rate = {learning_rate}")
    print(f"threshold = {threshold}")
    print(f"Best accuracy: {best_accuracy:.2f}")
    print(f"Best Normalized accuracy: {best_normalized_accuracy:.2f}")
    print(f"Best Pair accuracy: {best_pair_accuracy:.2f}")
    print(f"Best F1: {best_f1:.2f}")


def load_rnn(model_path):
    model = RnnModel(input_size, hidden_size, output_size).cuda()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def predict_correct(model, steps):
    # Timeouts are skipped
    if len(steps) == 0:
        return 0

    assert all(step["state"] in states for step in steps)
    X = [[1.0 if state == step["state"] else 0.0 for state in states] for step in steps]
    X = torch.tensor(X).unsqueeze(0).cuda()
    with torch.no_grad():
        output = model(X)
        return nn.Sigmoid()(output).item()


if __name__ == "__main__":

    train_filepath = (
        f"results/trace_{dataset}_train_{sample_count}_{reasoning_model_name}.json"
    )
    with open(train_filepath, "r") as file:
        train_data = json.load(file)

    problems = []
    for problem in train_data:
        problems.extend(
            [
                answer
                for answer in problem["verified_answers"]
                # Skip timeouts
                if not isinstance(answer, str)
            ]
        )

    train_rnn(problems)
