from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_groq import ChatGroq
import os
from memory import ConversationMemory

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

class ChatGroqFactory:
    @staticmethod
    def create_chat_groq(temperature=0.7, model_name="llama3-70b-8192"):
        return ChatGroq(groq_api_key=GROQ_API_KEY, temperature=temperature, model_name=model_name)

def chat_with_groq(self, system_prompt, user_prompt, chat_instance=None, memory=None, tools=None, max_tokens=1024):
    chat_instance = chat_instance or ChatGroqFactory.create_chat_groq()
    memory = memory or ConversationMemory()
    memory.save_context("user", user_prompt)

    remaining_tokens = max_tokens

    while True:
        relevant_messages = memory.retrieve_relevant_messages(user_prompt, memory)
        history = relevant_messages + [{"role": "user", "content": user_prompt}]

        messages = [SystemMessagePromptTemplate.from_template(system_prompt)] + [
            HumanMessagePromptTemplate.from_template(msg["content"]) if msg["role"] == "user" else
            SystemMessagePromptTemplate.from_template(msg["content"]) for msg in history
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        formatted_prompt = prompt.format_prompt()

        if len(formatted_prompt.split()) > remaining_tokens:
            break

        response = chat_instance.invoke(formatted_prompt, max_tokens=remaining_tokens)
        response_content = response.content
        remaining_tokens -= len(response_content.split())

        memory.save_context("assistant", response_content)

        if remaining_tokens <= 0:
            break

    return response_content