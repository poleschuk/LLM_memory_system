from config.LLM_config import chat, MODEL_NAME

class compressor():
    def __init__(self, text, client):
        self.text = text
        self.client = client

    def compress_with_llm(self, text):
        prompt = f"Summarize this memory in 1 short sentence:\n{text}"
        messages_contain = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        ai_reply = chat(self.client, MODEL_NAME, messages_contain)
        return ai_reply


    # def compress_user(self, text):
    #     if "I" in text:
    #         return text.replace("I", self.user)
    #     return text


    def compress(self):
        compressed_text = self.compress_with_llm(self.text)
        return compressed_text