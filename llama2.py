from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q4_0.gguf")


def get_prompt(instruction: str) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the questions in a short and concise way."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    print(prompt)
    return prompt


# def get_prompt(instruction: str, history: list[str] | None = None) -> str:

question = "Which city is the capital of philippines?"


for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()


# print(prompt + llm(prompt))
