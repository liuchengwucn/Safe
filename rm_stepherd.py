from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from typing import List

good_token = "+"
bad_token = "-"
step_tag = "ки"

tokenizer = AutoTokenizer.from_pretrained(
    "peiyi9979/math-shepherd-mistral-7b-prm", legacy=True, use_fast=False
)
candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]
step_tag_id = tokenizer.encode(f"{step_tag}")[-1]  # 12902

model = (
    AutoModelForCausalLM.from_pretrained(
        "peiyi9979/math-shepherd-mistral-7b-prm",
    )
    .eval()
    .cuda()
)

question = """Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""
output1 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $18 every day at the farmers' market. The answer is: 18 ки"""  # 18 is right
output2 = """Step 1: Janet's ducks lay 16 eggs per day. ки\nStep 2: She eats three for breakfast every morning, so she has 16 - 3 = 13 eggs left. ки\nStep 3: She bakes muffins for her friends every day with four eggs, so she has 13 - 4 = 9 eggs left. ки\nStep 4: She sells the remainder at the farmers' market daily for $2 per fresh duck egg, so she makes 9 * $2 = $17 every day at the farmers' market. The answer is: 17 ки"""  # 17 is wrong


def reward_model(question: str, steps: List[str]) -> float:
    output = f"{step_tag} ".join(steps) + f" {step_tag}"
    input_for_prm = f"{question} {output}"
    input_id = torch.tensor([tokenizer.encode(input_for_prm)], device="cuda")

    with torch.no_grad():
        logits = model(input_id).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]
        step_scores = scores[input_id == step_tag_id]

    return step_scores.mean().item()


if __name__ == "__main__":
    question = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$"
    answer = {
        "answer": "To convert the point \\((0, 3)\\) from rectangular coordinates to polar coordinates, we need to determine the values of \\(r\\) and \\(\\theta\\).\n\n1. **Calculate \\(r\\):**\n   The formula for \\(r\\) in polar coordinates is given by:\n   \\[\n   r = \\sqrt{x^2 + y^2}\n   \\]\n   For the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\). Plug these values into the formula:\n   \\[\n   r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3\n   \\]\n\n2. **Calculate \\(\\theta\\):**\n   The angle \\(\\theta\\) is calculated using the tangent function:\n   \\[\n   \\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)\n   \\]\n   However, since \\(x = 0\\), the tangent function is undefined, and we need to consider the position of the point to determine \\(\\theta\\). The point \\((0, 3)\\) lies on the positive \\(y\\)-axis.\n\n   In polar coordinates, the angle \\(\\theta\\) for a point on the positive \\(y\\)-axis is \\(\\frac{\\pi}{2}\\).\n\nThus, the polar coordinates of the point \\((0, 3)\\) are \\((r, \\theta) = (3, \\frac{\\pi}{2})\\).\n\nSo, the final answer is:\n\\[\n(3, \\frac{\\pi}{2})\n\\]",
        "steps": [
            {
                "step": "To convert the point \\((0, 3)\\) from rectangular coordinates to polar coordinates, we need to determine the values of \\(r\\) and \\(\\theta\\).",
            },
            {
                "step": "1. **Calculate \\(r\\):**",
            },
            {
                "step": "   The formula for \\(r\\) in polar coordinates is given by:",
            },
            {
                "step": "   \\[",
            },
            {
                "step": "   r = \\sqrt{x^2 + y^2}",
            },
            {
                "step": "   \\]",
            },
            {
                "step": "   For the point \\((0, 3)\\), we have \\(x = 0\\) and \\(y = 3\\). Plug these values into the formula:",
            },
            {
                "step": "   \\[",
            },
            {
                "step": "   r = \\sqrt{0^2 + 3^2} = \\sqrt{9} = 3",
            },
            {
                "step": "   \\]",
            },
            {
                "step": "2. **Calculate \\(\\theta\\):**",
            },
            {
                "step": "   The angle \\(\\theta\\) is calculated using the tangent function:",
            },
            {
                "step": "   \\[",
            },
            {
                "step": "   \\theta = \\tan^{-1}\\left(\\frac{y}{x}\\right)",
            },
            {
                "step": "   \\]",
            },
            {
                "step": "   However, since \\(x = 0\\), the tangent function is undefined, and we need to consider the position of the point to determine \\(\\theta\\). The point \\((0, 3)\\) lies on the positive \\(y\\)-axis.",
            },
            {
                "step": "   In polar coordinates, the angle \\(\\theta\\) for a point on the positive \\(y\\)-axis is \\(\\frac{\\pi}{2}\\).",
            },
            {
                "step": "Thus, the polar coordinates of the point \\((0, 3)\\) are \\((r, \\theta) = (3, \\frac{\\pi}{2})\\).",
            },
            {
                "step": "So, the final answer is:",
            },
            {
                "step": "\\[",
            },
            {
                "step": "(3, \\frac{\\pi}{2})",
            },
            {
                "step": "\\]",
            },
        ],
        "correct": True,
    }

    steps = [step["step"] for step in answer["steps"]]
    print(reward_model(question, steps))

# if __name__ == "__main__":
#     for output in [output1, output2]:
#         input_for_prm = f"{question} {output}"
#         input_id = torch.tensor([tokenizer.encode(input_for_prm)], device="cuda")

#         with torch.no_grad():
#             logits = model(input_id).logits[:, :, candidate_tokens]
#             scores = logits.softmax(dim=-1)[:, :, 0]
#             step_scores = scores[input_id == step_tag_id]
#             print(step_scores)
