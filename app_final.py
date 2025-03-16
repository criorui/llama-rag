# %%writefile app.py
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
from transformers import AutoTokenizer
import os
from huggingface_hub import InferenceClient

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"


Settings.llm = None
Settings.embed_model = HuggingFaceEmbedding(
    model_name="Alibaba-NLP/gte-multilingual-base",
    # pooling="mean",
    max_length=128,
    query_instruction="query: ",
    text_instruction="passage: ",
    trust_remote_code=True,
    device="cpu"
)


model_id = 'LenguajeNaturalAI/leniachat-qwen2-1.5B-v0'
client_url = "https://gdy55p88b036co-8080.proxy.runpod.net"

tokenizer=AutoTokenizer.from_pretrained(model_id)

DEFAULT_SYSTEM_MSSG = """Eres un asistente que responde preguntas al usuario y resuelve las instrucciones que este te plantea.
Tu tarea es resultar de ayuda para el usuario, siendo siempre amable, considerado, empático y divertido.
En caso de que se te proporcione información o contextos adicionales, utiliza únicamente la información del contexto o de la documentación para responder a las cuestiones que se te plantean.
"""

class ChatBot:
    def __init__(self, tokenizer, sampling_params: dict =None):
        self.sampling_params = sampling_params or dict(temperature=0.3, top_p=0.8, repetition_penalty=1.05, max_new_tokens=1024)
        self.tokenizer = tokenizer
        self.client = InferenceClient(client_url)
    
    def answer(self, message: str, context: str, chat_history: list[dict]):
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_MSSG},
            {"role": "input", "content": context},
        ]
        messages.extend(chat_history)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = ""
        for token in self.client.text_generation(prompt, **self.sampling_params, stream=True):
            response += token
            if token in ["<|im_end|>", "<|endoftext|>"]:
                break
            yield token
        


chatbot = ChatBot(tokenizer=tokenizer)
st.set_page_config(
    page_title="App de RAG",
    page_icon="📄",
    layout="wide",
)


storage_context = StorageContext.from_defaults(persist_dir="./storage_iso27001")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(streaming=False, similarity_top_k=1)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompts = []
# React to user input
if prompt := st.chat_input("Empieza a hablar con chatnatural"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    prompts.append(prompt)
    prompt_retrieve = "\n".join(prompts[::-1])
    nodes_retrieved = query_engine.retrieve(prompt_retrieve)
    context = "\n<doc>".join([node.text for node in nodes_retrieved])

    # Display assistant response in chat message container
    chatbot.chat_history = st.session_state.messages
    with st.chat_message("assistant"):
        response = st.write(chatbot.answer(prompt, context, st.session_state.messages))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})