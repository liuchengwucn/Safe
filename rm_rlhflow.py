from typing import List
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch

model = (
    AutoModelForCausalLM.from_pretrained(
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data", torch_dtype=torch.bfloat16
    )
    .to("cuda")
    .eval()
)
tokenizer = AutoTokenizer.from_pretrained("RLHFlow/Llama3.1-8B-PRM-Deepseek-Data")

plus_tag_id = tokenizer.encode("+")[-1]
minus_tag_id = tokenizer.encode("-")[-1]
candidate_tokens = [plus_tag_id, minus_tag_id]


def reward_model(
    question: str,
    steps: List[str],
) -> float:
    messages = []
    step_scores = []
    if len(steps) == 0:
        return 0.0
    for i in range(len(steps)):
        if i == 0:
            text = f"{question} {steps[i]}"
        else:
            text = steps[i]

        messages.append({"role": "assistant", "content": "+"})
        messages.append({"role": "user", "content": text})

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda"
        )
        with torch.no_grad():
            logits = model(input_ids).logits[
                :, -3, candidate_tokens
            ]  # simple version, the +/- is predicted by the '-3' position
            scores = logits.softmax(dim=-1)[:, 0]  # 0 means the prob of + (1 mean -)
            score = scores[0].detach().to("cpu", dtype=torch.float32).item()
        step_scores.append(score)

    return sum(step_scores) / len(step_scores)


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
