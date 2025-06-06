import os
import openai
import backoff
from typing import List
from functools import partial

from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "")
use_vllm = True

client = openai.OpenAI(api_key=api_key, base_url=base_url)
vllm_client = openai.OpenAI(base_url="http://reasoning-model:8000/v1")


@backoff.on_exception(backoff.expo, openai.APIError)
def completions_with_backoff(model: str, **kwargs):
    if not model.startswith("gpt"):
        return vllm_client.chat.completions.create(model=model, **kwargs)
    return client.chat.completions.create(model=model, **kwargs)


def gpt(
    prompt,
    model_name="gpt-4o-mini-2024-07-18",
    temperature=0.7,
    max_tokens=1000,
    n=1,
    top_p=None,
    stop=[],
) -> List[str]:
    if model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        if len(prompt) > 6000:
            prompt = prompt[-6000:]
        max_tokens = 8192 - len(prompt)

    global completion_tokens, prompt_tokens
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    while n > 0:
        cnt = min(n, 16)
        n -= cnt
        res = completions_with_backoff(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=cnt,
            stop=stop,
            top_p=top_p,
        )
        outputs.extend([choice.message.content for choice in res.choices])
    return outputs


gpt4o = partial(gpt, model_name="gpt-4o-2024-08-06", max_tokens=8192)
gpt4o_mini = partial(gpt, model_name="gpt-4o-mini-2024-07-18", max_tokens=8192)
llama = partial(gpt, model_name="meta-llama/Llama-3.1-8B-Instruct", max_tokens=8192)
llama30 = partial(
    gpt, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_tokens=8192
)


def deepseek_math(prompt: str, max_tokens=2000, **kwargs):
    # context length for deepseek-math-7b-instruct is 4096
    max_prompt_length = 4096 - max_tokens
    if len(prompt) > max_prompt_length:
        prompt = prompt[-max_prompt_length:]
    return gpt(prompt, model_name="deepseek-ai/deepseek-math-7b-instruct", **kwargs)


MODEL = gpt4o_mini

if __name__ == "__main__":
    print(vllm_client.models.list())
    print(gpt4o("Hello! Who are you?", n=1))
    print(gpt4o_mini("Hello! Who are you?", n=1))
    # print(deepseek_math("Hello! Who are you?", n=1))
    print(
        llama(
            "Hello! Who are you?",
        )
    )
