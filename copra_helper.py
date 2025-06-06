import os
import json
import shutil
from typing import Tuple

COPRA_PATH = "./copra"

with open("prompts/lean_import.lean", "r") as f:
    lean_import = f.read()


def find_file(filename, search_path) -> str:
    for dirpath, dirnames, files in os.walk(search_path):
        if filename in files:
            return os.path.join(dirpath, filename)
    return ""


def prove_and_check(statement: str) -> Tuple[bool, str | None]:
    """
    Proof a statement using COPRA
    """
    # Write the statement to a file
    with open(f"{COPRA_PATH}/data/test/lean4_proj/Lean4Proj/Temp.lean", "w") as f:
        f.write(lean_import)
        f.write(statement)

    # Delete the old output file
    shutil.rmtree(f"{COPRA_PATH}/.log/", ignore_errors=True)
    shutil.rmtree(f"{COPRA_PATH}/outputs/", ignore_errors=True)

    # Run COPRA
    os.system(
        f"cd {COPRA_PATH} && python ./src/main/eval_benchmark.py --config-name lean4_simple_experiment"
    )

    # Read the result
    with open(find_file("proof_results.json", f"{COPRA_PATH}/.log/proofs/"), "r") as f:
        result = json.load(f)

    # Return the result
    # The result looks like this:
    {
        "path": ".log/proofs/eval_driver/dfs/simple_benchmark_lean4/20241114-183946/proof_results.json",
        "theorem_map": {
            "data/test/lean4_proj/Lean4Proj/Temp.lean": {
                "test": {
                    "proof_file": None,
                    "proof_found": True,
                    "lemma_name": "theorem test\n(b x: ℝ)\n(h₀: x - 5 = 2 * x - b)\n(h₁: x = -2):\n(b = 3)  :=",
                    "proof_steps": [
                        {
                            "proof_id": None,
                            "all_useful_defns_theorems": [],
                            "goal_description": None,
                            "start_goals": [],
                            "end_goals": [],
                            "proof_steps": ["rw [h₁] at h₀"],
                            "simplified_goals": [],
                            "addition_state_info": {},
                        },
                        {
                            "proof_id": None,
                            "all_useful_defns_theorems": [],
                            "goal_description": None,
                            "start_goals": [],
                            "end_goals": [],
                            "proof_steps": ["linear_combination h₀"],
                            "simplified_goals": [],
                            "addition_state_info": {},
                        },
                    ],
                    "proof_time_in_secs": 4.091809272766113,
                    "inferences_taken": 2,
                    "possible_failed_paths": -1,
                    "num_of_backtracks": 0,
                    "is_timeout": False,
                    "is_inference_exhausted": False,
                    "longest_success_path": -1,
                    "additional_info": {
                        "queries": 2,
                        "attempt_idx": 3,
                        "total_queries": 16,
                    },
                    "language": "LEAN4",
                }
            }
        },
    }

    results = result["theorem_map"]["data/test/lean4_proj/Lean4Proj/Temp.lean"]
    theorem_name = list(results.keys())[0]
    proved = results[theorem_name]["proof_found"]

    # Extract theorem proof
    proof = None
    if proved:
        proof_steps = results[theorem_name]["proof_steps"]
        proof = "\n".join(sum((step["proof_steps"] for step in proof_steps), []))

    return proved, proof


if __name__ == "__main__":
    print(
        prove_and_check(
            """theorem test
  (b x: ℝ)
  (h₀: x - 5 = 2 * x - b)
  (h₁: x = -2):
  (b = 3) := by sorry
"""
        )
    )
