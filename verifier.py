from typing import List
from formalize import formalize
import re

from utils import call_scheduler_with_timeout


class Verifier:
    def formalize(
        self, question: str, current_state: List[str], step: str
    ) -> str | None:
        output = formalize(question, current_state, step)
        # The output may be surrounded by ``` or ```lean
        if output is None:
            return None
        output = (
            re.sub(r"```(Lean4|Lean3|Lean|lean4|lean3|lean)", "", output)
            .replace("```", "")
            .strip()
        )
        return output

    def lean_verifier(self, formal_provement: str) -> bool:
        with open("prompts/lean_import.lean") as f:
            lean_import = f.read()

        scheduler_input = [
            dict(code=lean_import + formal_provement, ast=False, tactics=False)
        ]
        outputs_list = call_scheduler_with_timeout(scheduler_input)

        # Timeout
        if len(outputs_list) == 0:
            return False
        return outputs_list[0]["pass"] and (len(outputs_list[0]["sorries"]) == 1)


if __name__ == "__main__":
    v = Verifier()

    question = "What is the sum of 1 and 2?"
    current_state = [
        "To find the sum of 1 and 2, let's break it down step by step:",
        "1. **Identify the numbers to be added**: The numbers are 1 and 2.",
        "2. **Add the numbers together**:  ( 1 + 2 = 3 )",
    ]
    step = "3. **Verify the result**: Counting forward from 1 by 2 units gives us 3 (1 → 2, 2 → 3)."
    formal_provement = v.formalize(question, current_state, step)

    print(f"Formal Provement: {formal_provement}")
    print(f"Lean Verifier Result: {v.lean_verifier(formal_provement)}")
