# local_llm_prompt.py
import os
from llama_cpp import Llama
from starlette.concurrency import run_in_threadpool

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=1024,
    n_threads=os.cpu_count(),
    n_batch=1024,
    use_mlock=True,
    verbose=False,
)


def ask_model(prompt: str) -> str:
    full_prompt = f"<s>[INST] {prompt} [/INST]"
    output = llm(full_prompt, max_tokens=128, temperature=0.7)
    return output["choices"][0]["text"].strip()


async def ask_model_async(prompt: str) -> str:
    return await run_in_threadpool(ask_model, prompt)
