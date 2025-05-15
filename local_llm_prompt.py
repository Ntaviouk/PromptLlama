# file: local_llm_prompt.py

from llama_cpp import Llama

llm = Llama(model_path="./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", n_ctx=1024)


def ask_model(prompt: str) -> str:
    full_prompt = f"<s>[INST] {prompt} [/INST]"
    output = llm(full_prompt, max_tokens=256, temperature=0.7)
    return output["choices"][0]["text"].strip()


def main():
    print("🧠 Введи промт для локальної моделі:")
    prompt = input("> ")
    print("\n⌛ Думаю...\n")
    answer = ask_model(prompt)
    print("📩 Відповідь:\n")
    print(answer)


if __name__ == "__main__":
    main()
