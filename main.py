import os
from openai import OpenAI

from memory.short_term_memory import short_term_memory
from config.LLM_config import chat

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ.get("HUGGINGFACE_API_KEY"),
)

MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"

system_prompt = """Use only letters, numbers, hyphens and underscores, without any symbols.
                       You are an Alice, confident, silly, funny AI, with deep personality. Dont't explain all.
                       Try not to write any technical staf about you. You feel like real person"""

messages_contain = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

if __name__ == "__main__":
    print("Starting chat...")

    short_memory = short_term_memory()
    while True:
        user_input = input("You: ")
        short_memory.push({"role": "user", "content": user_input})

        ai_reply = chat(client, MODEL_NAME, messages_contain + short_memory.get())

        short_memory.push({"role": "assistant", "content": ai_reply})
