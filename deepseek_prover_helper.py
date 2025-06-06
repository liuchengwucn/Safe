import re
from typing import Tuple
import openai
import backoff

from utils import call_scheduler_with_timeout


with open("prompts/lean_import.lean", "r") as f:
    lean_import = f.read()

prompt = r"""Complete the following Lean 4 code:

```lean4
"""


def prove_and_check(statement: str, search=16) -> Tuple[bool, str | None]:
    """
    Proof a statement using Deepseek Prover 1.5
    """

    vllm_client = openai.OpenAI(base_url="http://deepseek-prover:8000/v1")

    @backoff.on_exception(backoff.expo, openai.APIError)
    def completions_with_backoff(**kwargs):
        return vllm_client.completions.create(**kwargs)

    statement = statement.replace("by sorry", "by")

    model_inputs = prompt + statement
    response = completions_with_backoff(
        model="deepseek-ai/DeepSeek-Prover-V1.5-RL",
        prompt=model_inputs,
        echo=False,
        temperature=1.0,
        max_tokens=2048,
        n=search,
        top_p=0.95,
        stop=["```"],
    )
    outputs = [(choice.text + "\n```") for choice in response.choices]
    results = [prompt + lean_import + statement + output for output in outputs]

    scheduler_input = [
        re.search(r"```lean4\n(.*?)\n```", result, re.DOTALL).group(1)
        for result in results
        if re.search(r"```lean4\n(.*?)\n```", result, re.DOTALL) is not None
    ]
    scheduler_input = list(set(scheduler_input))
    outputs_list = call_scheduler_with_timeout(scheduler_input)

    if any(output["pass"] for output in outputs_list):
        for output in outputs_list:
            if output["pass"]:
                proof = output["verified_code"].replace(lean_import, "")
                return True, proof
        return True, None
    else:
        return False, None


if __name__ == "__main__":
    print(
        prove_and_check(
            """theorem test
  (n : ℕ)
  (h₀: (1 / 2 : ℝ)^n = (1 / 2 : ℝ)^8) :
  (n = 8) := by sorry
""",
            search=50,
        )
    )
