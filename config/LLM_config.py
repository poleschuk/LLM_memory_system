from openai import OpenAI

MODEL_NAME = "moonshotai/Kimi-K2-Instruct-0905"

def chat(client: OpenAI, model_name: str, messages_contain):
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages_contain,
        max_tokens=500,
        temperature=0.2,
        top_p=1.0
    )
    reply = completion.choices[0].message.content
    return reply
