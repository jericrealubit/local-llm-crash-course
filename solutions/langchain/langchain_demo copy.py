from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="context", input_key="instruction")

llm_llama2 = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q4_K_M.gguf",
    model_type="llama2",
    max_new_tokens=20
)

llm_orca = CTransformers(
    model="zoltanctoth/orca_mini_3B-GGUF",
    model_file="orca-mini-3b.q4_0.gguf",
    model_type="llama2",
    max_new_tokens=20,
)

prompt_template = """
### System:\nYou are an AI assistant that gives helpful answers. You answer the question in a short and concise way.
Context: {context}

### User: {instruction}

### Response: """

prompt = PromptTemplate(template=prompt_template, input_variables=["instruction", "context"])

chain = LLMChain(llm=llm_llama2, prompt=prompt, verbose=True, memory=memory)


def chat_interface(user_input: str):
    cleaned_input = user_input.strip().lower()
    if cleaned_input == "forget everything":
        memory.clear()
        return "Uh oh, I've just forgotten our conversation history"

    elif cleaned_input == "use llama2":
        chain.llm = llm_llama2
        return "Model changed to Llama"

    elif cleaned_input == "use orca":
        chain.llm = llm_orca
        return "Model changed to Orca"

    else:
        return chain.invoke({"instruction": user_input})


print(chat_interface("Which city is the capital of India?"))
print(chat_interface("Which has the same functionality in the US?"))
print(chat_interface("Forget everything"))
print(chat_interface("Which has the same functionality in the Philippines?"))
print(chat_interface("Use Orca"))
print(chat_interface("Which has the same functionality in the China?"))
print(chat_interface("Use Llama2"))
print(chat_interface("Which has the same functionality in the New Zealand?"))
