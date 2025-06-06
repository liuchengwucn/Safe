from typing import Dict, List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ArmoRMPipeline:
    def __init__(
        self,
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        truncation=True,
        trust_remote_code=False,
        max_length=4096,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


rm = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)


def reward_model(
    question: str,
    steps: List[str],
) -> float:
    conv = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": "\n".join(steps)},
    ]

    return rm(conv)["score"]


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
